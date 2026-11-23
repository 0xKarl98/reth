//! Storage lock utils.

#![cfg_attr(feature = "disable-lock", allow(dead_code))]

use reth_fs_util::{self as fs, FsPathError};
use reth_storage_errors::lockfile::StorageLockError;
use same_file::Handle;
use std::{
    ffi::OsStr,
    fs::File,
    io::Write,
    path::{Path, PathBuf},
    process,
    sync::{Arc, OnceLock},
};
use sysinfo::{ProcessRefreshKind, RefreshKind, System};

/// File lock name.
const LOCKFILE_NAME: &str = "lock";

/// Stable inode for shared writer and exclusive cleanup locks.
const GUARD_FILE_NAME: &str = ".reth-storage-lock";

/// Prevents a writer from reopening stores after a partially completed destructive reset.
const DROP_MARKER: &str = ".reth-drop-in-progress";

/// A file lock for a storage directory to ensure exclusive read-write access across different
/// processes.
///
/// This lock stores the PID of the process holding it and is released (deleted) on a graceful
/// shutdown. On resuming from a crash, the stored PID helps verify that no other process holds the
/// lock.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StorageLock(Arc<StorageLockInner>);

impl StorageLock {
    /// Tries to acquire a write lock on the target directory, returning [`StorageLockError`] if
    /// unsuccessful.
    ///
    /// Note: In-process exclusivity is not on scope. If called from the same process (or another
    /// with the same PID), it will succeed.
    pub fn try_acquire(path: &Path) -> Result<Self, StorageLockError> {
        #[cfg(feature = "disable-lock")]
        {
            // Too expensive for ef-tests to write/read lock to/from disk.
            Ok(Self(Arc::new(StorageLockInner {
                directory: path.into(),
                guard: None,
                exclusive: false,
            })))
        }

        #[cfg(not(feature = "disable-lock"))]
        Self::try_acquire_file_lock(path)
    }

    /// Returns whether a directory entry is part of the storage writer lock.
    pub fn is_lock_file(name: &OsStr) -> bool {
        name == LOCKFILE_NAME || name == GUARD_FILE_NAME
    }

    #[cfg(any(test, not(feature = "disable-lock")))]
    fn try_acquire_file_lock(path: &Path) -> Result<Self, StorageLockError> {
        StorageLockInner::acquire(path, false).map(|inner| Self(Arc::new(inner)))
    }
}

/// Exclusive storage guard for destructive maintenance. Ordinary writers can reacquire their
/// locks within one process, but cannot share a directory with a reset.
#[derive(Debug)]
pub struct StorageDropLock(StorageLockInner);

impl StorageDropLock {
    /// Locks an existing or missing directory without opening the store or validating its data.
    /// Returns an error if a writer or another reset holds it.
    pub fn try_acquire(path: &Path) -> Result<Self, StorageLockError> {
        StorageLockInner::acquire(path, true).map(Self)
    }

    /// Returns the recorded targets of an interrupted reset, or `None` before cleanup starts.
    pub fn drop_marker(&self) -> fs::Result<Option<Vec<u8>>> {
        match fs::read(self.0.directory.join(DROP_MARKER)) {
            Ok(bytes) => Ok(Some(bytes)),
            Err(FsPathError::Read { source, .. })
                if source.kind() == std::io::ErrorKind::NotFound =>
            {
                Ok(None)
            }
            Err(err) => Err(err),
        }
    }

    /// Atomically records the confirmed targets before any store is removed. Writers remain
    /// excluded until [`Self::finish_drop`] removes the marker.
    pub fn begin_drop(&self, targets: &[u8]) -> fs::Result<()> {
        fs::atomic_write_file(&self.0.directory.join(DROP_MARKER), |file| file.write_all(targets))
    }

    /// Allows writers to reopen the database after all its stores have been cleared.
    pub fn finish_drop(&self) -> fs::Result<()> {
        fs::remove_file(self.0.directory.join(DROP_MARKER))
    }

    /// Destructively removes the entries in the locked directory, keeping the writer lock until
    /// the guard is dropped. An interrupted removal can be retried with a new guard.
    pub fn clear_directory(&self) -> fs::Result<()> {
        let directory = &self.0.directory;
        for entry in fs::read_dir(directory)? {
            let entry = entry.map_err(|err| FsPathError::read_dir(err, directory))?;
            let name = entry.file_name();
            if StorageLock::is_lock_file(&name) || name == DROP_MARKER {
                continue
            }
            let path = entry.path();
            if entry.file_type().map_err(|err| FsPathError::metadata(err, &path))?.is_dir() {
                fs::remove_dir_all(path)?;
            } else {
                fs::remove_file(path)?;
            }
        }
        Ok(())
    }
}

impl Drop for StorageLockInner {
    fn drop(&mut self) {
        // The lockfile is not created in disable-lock mode, so we don't need to delete it.
        if self.guard.is_some() {
            let file_path = self.directory.join(LOCKFILE_NAME);
            if file_path.exists() {
                if let Ok(Some(process_uid)) = ProcessUID::parse(&file_path) {
                    // Only remove if the lock file belongs to our process
                    if process_uid.pid == process::id() as usize {
                        if let Err(err) = reth_fs_util::remove_file(file_path) {
                            tracing::error!(%err, "Failed to delete lock file");
                        }
                    } else {
                        tracing::warn!(
                            "Lock file belongs to different process (PID: {}), not removing",
                            process_uid.pid
                        );
                    }
                } else {
                    // If we can't parse the lock file, still try to remove it
                    // as it might be corrupted or from a previous run
                    if let Err(err) = reth_fs_util::remove_file(file_path) {
                        tracing::error!(%err, "Failed to delete lock file");
                    }
                }
            }
            if self.exclusive {
                let path = self.directory.join(GUARD_FILE_NAME);
                if let Err(err) = fs::remove_file(&path) {
                    tracing::error!(%err, "Failed to remove the cleanup lock file");
                }
            }
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
struct StorageLockInner {
    directory: PathBuf,
    /// Keeps a reset exclusive even while a writer is starting and publishing its PID.
    guard: Option<Handle>,
    exclusive: bool,
}

impl StorageLockInner {
    /// Acquire a file write lock.
    fn acquire(path: &Path, exclusive: bool) -> Result<Self, StorageLockError> {
        fs::create_dir_all(path).map_err(StorageLockError::other)?;
        let guard_path = path.join(GUARD_FILE_NAME);
        let guard = File::options()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&guard_path)
            .map_err(StorageLockError::other)?;
        let guard = Handle::from_file(guard).map_err(StorageLockError::other)?;
        if exclusive { guard.as_file().try_lock() } else { guard.as_file().try_lock_shared() }
            .map_err(StorageLockError::other)?;

        // The previous reset may have unlinked this inode between open and try_lock.
        if guard != Handle::from_path(&guard_path).map_err(StorageLockError::other)? {
            return Err(StorageLockError::Other(format!("Storage lock changed at {guard_path:?}")))
        }

        if !exclusive && path.join(DROP_MARKER).try_exists().map_err(StorageLockError::other)? {
            return Err(StorageLockError::Other(format!(
                "Unfinished database drop at {}. Retry reth db drop before starting a writer",
                path.display()
            )))
        }

        let file_path = path.join(LOCKFILE_NAME);
        if let Some(process_lock) = ProcessUID::parse(&file_path)? &&
            process_lock.pid != (process::id() as usize) &&
            process_lock.is_active()
        {
            tracing::error!(
                target: "reth::db::lockfile",
                path = ?file_path,
                pid = process_lock.pid,
                start_time = process_lock.start_time,
                "Storage lock already taken."
            );
            return Err(StorageLockError::Taken(process_lock.pid))
        }

        ProcessUID::own().write(&file_path)?;
        Ok(Self { directory: path.into(), guard: Some(guard), exclusive })
    }
}

#[derive(Clone, Debug)]
struct ProcessUID {
    /// OS process identifier
    pid: usize,
    /// Process start time
    start_time: u64,
}

impl ProcessUID {
    /// Creates [`Self`] for the provided PID.
    fn new(pid: usize) -> Option<Self> {
        let mut system = System::new();
        let pid2 = sysinfo::Pid::from(pid);
        system.refresh_processes_specifics(
            sysinfo::ProcessesToUpdate::Some(&[pid2]),
            true,
            ProcessRefreshKind::nothing(),
        );
        system.process(pid2).map(|process| Self { pid, start_time: process.start_time() })
    }

    /// Creates [`Self`] from own process.
    fn own() -> Self {
        static CACHE: OnceLock<ProcessUID> = OnceLock::new();
        CACHE.get_or_init(|| Self::new(process::id() as usize).expect("own process")).clone()
    }

    /// Parses [`Self`] from a file.
    fn parse(path: &Path) -> Result<Option<Self>, StorageLockError> {
        if path.exists() &&
            let Ok(contents) = reth_fs_util::read_to_string(path)
        {
            let mut lines = contents.lines();
            if let (Some(Ok(pid)), Some(Ok(start_time))) = (
                lines.next().map(str::trim).map(str::parse),
                lines.next().map(str::trim).map(str::parse),
            ) {
                return Ok(Some(Self { pid, start_time }));
            }
        }
        Ok(None)
    }

    /// Whether a process with this `pid` and `start_time` exists.
    fn is_active(&self) -> bool {
        System::new_with_specifics(
            RefreshKind::nothing().with_processes(ProcessRefreshKind::nothing()),
        )
        .process(self.pid.into())
        .is_some_and(|p| p.start_time() == self.start_time)
    }

    /// Writes `pid` and `start_time` to a file.
    fn write(&self, path: &Path) -> Result<(), StorageLockError> {
        reth_fs_util::write(path, format!("{}\n{}", self.pid, self.start_time))
            .map_err(StorageLockError::other)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Mutex, MutexGuard, OnceLock};

    // helper to ensure some tests are run serially
    static SERIAL: OnceLock<Mutex<()>> = OnceLock::new();

    fn serial_lock() -> MutexGuard<'static, ()> {
        SERIAL.get_or_init(|| Mutex::new(())).lock().unwrap()
    }

    #[test]
    fn test_lock() {
        let _guard = serial_lock();

        let temp_dir = tempfile::tempdir().unwrap();

        let lock = StorageLock::try_acquire_file_lock(temp_dir.path()).unwrap();

        // Same process can re-acquire the lock
        assert_eq!(Ok(lock.clone()), StorageLock::try_acquire_file_lock(temp_dir.path()));

        // A lock of a non existent PID can be acquired.
        let lock_file = temp_dir.path().join(LOCKFILE_NAME);
        let mut fake_pid = 1337;
        let system = System::new_all();
        while system.process(fake_pid.into()).is_some() {
            fake_pid += 1;
        }
        ProcessUID { pid: fake_pid, start_time: u64::MAX }.write(&lock_file).unwrap();
        assert_eq!(Ok(lock.clone()), StorageLock::try_acquire_file_lock(temp_dir.path()));

        let mut pid_1 = ProcessUID::new(1).unwrap();

        // If a parsed `ProcessUID` exists, the lock can NOT be acquired.
        pid_1.write(&lock_file).unwrap();
        assert_eq!(
            Err(StorageLockError::Taken(1)),
            StorageLock::try_acquire_file_lock(temp_dir.path())
        );

        // A lock of a different but existing PID can be acquired ONLY IF the start_time differs.
        pid_1.start_time += 1;
        pid_1.write(&lock_file).unwrap();
        assert_eq!(Ok(lock), StorageLock::try_acquire_file_lock(temp_dir.path()));
    }

    #[test]
    fn test_drop_lock() {
        let _guard = serial_lock();

        let temp_dir = tempfile::tempdir().unwrap();
        let lock_file = temp_dir.path().join(LOCKFILE_NAME);

        let lock = StorageLock::try_acquire_file_lock(temp_dir.path()).unwrap();

        assert!(lock_file.exists());
        drop(lock);
        assert!(!lock_file.exists());
    }

    #[test]
    fn reset_excludes_writers_through_cleanup() {
        let dir = tempfile::tempdir().unwrap();
        let writer = StorageLock::try_acquire_file_lock(dir.path()).unwrap();
        assert!(matches!(
            StorageDropLock::try_acquire(dir.path()),
            Err(StorageLockError::Other(_))
        ));
        drop(writer);
        let reset = StorageDropLock::try_acquire(dir.path()).unwrap();
        fs::write(dir.path().join("data"), b"data").unwrap();
        reset.clear_directory().unwrap();
        assert!(!dir.path().join("data").exists());
        assert!(matches!(
            StorageLock::try_acquire_file_lock(dir.path()),
            Err(StorageLockError::Other(_))
        ));
        assert!(matches!(
            StorageDropLock::try_acquire(dir.path()),
            Err(StorageLockError::Other(_))
        ));
        drop(reset);
        assert_eq!(fs::read_dir(dir.path()).unwrap().count(), 0);
        StorageLock::try_acquire_file_lock(dir.path()).unwrap();
    }
}
