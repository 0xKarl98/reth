//! Destructive cleanup of the configured database stores.

use crate::common::EnvironmentArgs;
use alloy_primitives::B256;
use eyre::{Result, WrapErr};
use reth_chainspec::EthChainSpec;
use reth_cli::chainspec::ChainSpecParser;
use reth_db::{
    lockfile::{StorageDropLock, StorageLock},
    open_db_read_only,
};
use reth_db_api::{database::Database, tables::HeaderNumbers, transaction::DbTx};
use reth_fs_util as fs;
use reth_node_builder::NodePrimitives;
use reth_node_core::dirs::{ChainPath, DataDirPath};
use reth_provider::{
    providers::{RocksDBProvider, StaticFileProviderBuilder},
    BlockHashReader,
};
use serde::Serialize;
use std::{
    ffi::OsStr,
    io::{self, Write},
    path::{self, Path, PathBuf},
};
use tracing::info;

pub(super) fn execute<P: NodePrimitives, C: ChainSpecParser<ChainSpec: EthChainSpec>>(
    env: &EnvironmentArgs<C>,
    data_dir: &ChainPath<DataDirPath>,
    force: bool,
) -> Result<()> {
    let db = path::absolute(data_dir.db())?;
    let rocks = path::absolute(data_dir.rocksdb())?;
    let static_files = path::absolute(data_dir.static_files())?;
    let wal = path::absolute(data_dir.exex_wal())?;

    if !force {
        print!(
            "Permanently delete data at these paths?\n  MDBX: {}\n  RocksDB (dedicated directory): {}\n  Static files: {}\n  ExEx WAL: {}\nThis cannot be undone. (y/N): ",
            db.display(), rocks.display(), static_files.display(), wal.display()
        );
        io::stdout().flush()?;
        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        if !input.trim().eq_ignore_ascii_case("y") {
            println!("Database drop aborted!");
            return Ok(())
        }
    }

    let db_lock = lock_directory(&db)?;
    let targets = serde_json::to_vec(&DropTargets {
        version: DROP_MARKER_VERSION,
        genesis: env.chain.genesis_hash(),
        rocksdb: resolved_target(&rocks, true)?.as_os_str(),
        static_files: resolved_target(&static_files, false)?.as_os_str(),
        wal: resolved_target(&wal, false)?.as_os_str(),
    })?;
    let marker = db_lock.drop_marker()?;
    let confirmed = marker.is_some();
    if let Some(previous) = marker {
        eyre::ensure!(
            previous == targets,
            "Unfinished database drop has different targets. Retry with the original chain and storage paths"
        );
    }
    let static_lock = lock_directory(&static_files)?;

    // Check directory ownership before removing anything, including when overrides resolve through
    // symlinks. Nested stores would let recursive cleanup unlink another store's writer lock.
    let mut roots = Vec::<PathBuf>::new();
    for path in [&db, &static_files, &rocks, &wal] {
        if path.try_exists()? || path.is_symlink() {
            let root = path
                .canonicalize()
                .wrap_err_with(|| format!("Could not resolve {}", path.display()))?;
            for other in &roots {
                eyre::ensure!(
                    !root.starts_with(other) && !other.starts_with(&root),
                    "Database directories overlap: {root:?} and {other:?}"
                );
            }
            roots.push(root);
        }
    }

    if !confirmed {
        verify_genesis::<P, _>(env, &db, &static_files)?;
    }
    // Reth disables WAL archival. Native destruction would also remove unrelated numbered
    // logs from this directory, so leave an existing archive for the operator to inspect.
    let archive = rocks.join("archive");
    if archive.try_exists()? {
        eyre::ensure!(
            fs::read_dir(&archive)?.next().transpose()?.is_none(),
            "Move the contents of {} before dropping RocksDB",
            archive.display()
        );
    }
    // CURRENT can already be gone on retry. Native destroy recognizes files by name and would
    // also delete a numbered application log in a directory that never contained RocksDB.
    let current = rocks.join("CURRENT");
    let exists = rocks.try_exists()?;
    if exists {
        let recognized =
            confirmed || (current.try_exists()? && RocksDBProvider::has_reth_tables(&rocks)?);
        // DestroyDB can report success if it cannot list the directory.
        for entry in fs::read_dir(&rocks)? {
            entry.map_err(|err| fs::FsPathError::read_dir(err, &rocks))?;
            eyre::ensure!(
                recognized,
                "Refusing to destroy an unrecognized RocksDB directory at {}",
                rocks.display()
            );
        }
    }
    info!(target: "reth::cli", ?db, ?rocks, ?static_files, ?wal, "Dropping database stores");
    if !confirmed {
        db_lock.begin_drop(&targets)?;
    }
    if exists {
        RocksDBProvider::destroy(&rocks)
            .wrap_err_with(|| format!("Could not destroy RocksDB at {}", rocks.display()))?;
        eyre::ensure!(
            !current.try_exists()?,
            "RocksDB cleanup left {} in place",
            current.display()
        );
    }
    if wal.try_exists()? {
        fs::remove_dir_all(&wal)?;
    }
    if static_files.is_symlink() {
        fs::remove_dir_all(&static_files)?;
        fs::create_dir_all(&static_files)?;
    } else {
        static_lock.clear_directory()?;
    }
    if db.is_symlink() {
        fs::remove_dir_all(&db)?;
        db_lock.finish_drop()?;
    } else {
        db_lock.clear_directory()?;
        db_lock.finish_drop()?;
        drop(db_lock);
        // A new writer can acquire the lock now. Never recursively remove its files.
        std::fs::remove_dir(&db).wrap_err_with(|| format!("Could not remove {}", db.display()))?;
    }
    Ok(())
}

fn lock_directory(path: &Path) -> Result<StorageDropLock> {
    // A root symlink is unlinked during cleanup. Release its lock through the original target.
    let lock = if path.is_symlink() {
        StorageDropLock::try_acquire(&path.canonicalize()?)
    } else {
        StorageDropLock::try_acquire(path)
    };
    lock.wrap_err_with(|| format!("Could not lock {}", path.display()))
}

/// Verify the chain independently in each populated store before deleting either one.
fn verify_genesis<P: NodePrimitives, C: ChainSpecParser<ChainSpec: EthChainSpec>>(
    env: &EnvironmentArgs<C>,
    db: &Path,
    static_files: &Path,
) -> Result<()> {
    let hash = env.chain.genesis_hash();
    let number = env.chain.genesis().number.unwrap_or_default();
    let static_hash = StaticFileProviderBuilder::read_only(static_files)
        .with_genesis_block_number(number)
        .build::<P>()?
        .block_hash(number)?;

    if db.join("mdbx.dat").try_exists()? {
        let store = open_db_read_only(db, env.db.database_args())?;
        let tx = store.tx()?;
        eyre::ensure!(
            tx.get::<HeaderNumbers>(hash)? == Some(number) ||
                (static_hash.is_some() && tx.entries::<HeaderNumbers>()? == 0),
            "Database does not contain the configured genesis {hash}"
        );
    }
    // An empty static store has no chain data. A populated one must identify itself even if
    // MDBX belongs to the requested chain or has an empty genesis index.
    if let Some(stored) = static_hash {
        eyre::ensure!(
            stored == hash,
            "Static files at {} do not contain the configured genesis {hash}",
            static_files.display()
        );
    } else {
        for entry in fs::read_dir(static_files)? {
            eyre::ensure!(
                StorageLock::is_lock_file(&entry?.file_name()),
                "Missing genesis in nonempty static files at {}",
                static_files.display()
            );
        }
    }
    Ok(())
}

/// Resolve the existing prefix so a target keeps the same identity after cleanup removes it.
/// `RocksDB` follows the final symlink, while static-file and WAL cleanup unlink it.
fn resolved_target(path: &Path, follow_final_symlink: bool) -> Result<PathBuf> {
    let resolve_from = if follow_final_symlink { path } else { path.parent().unwrap_or(path) };
    for ancestor in resolve_from.ancestors() {
        if ancestor.try_exists()? {
            let mut resolved = ancestor.canonicalize()?;
            resolved.extend(path.strip_prefix(ancestor)?.components());
            return Ok(resolved)
        }
    }
    eyre::bail!("Could not resolve drop target {}", path.display())
}

const DROP_MARKER_VERSION: u8 = 1;

/// Identifies the cleanup authorized by the operator even after genesis has been removed.
#[derive(Serialize)]
struct DropTargets<'a> {
    version: u8,
    genesis: B256,
    rocksdb: &'a OsStr,
    static_files: &'a OsStr,
    wal: &'a OsStr,
}
