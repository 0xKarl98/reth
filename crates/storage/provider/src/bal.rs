use alloy_primitives::{BlockHash, BlockNumber, Bytes};
use parking_lot::RwLock;
use reth_storage_api::BalStore;
use reth_storage_errors::provider::ProviderResult;
use std::{
    collections::{BTreeMap, HashMap},
    sync::Arc,
};

/// Default number of recent BALs kept in memory.
const DEFAULT_RECENT_BAL_CAPACITY: usize = 1024;

/// In-memory BAL store for recently validated blocks.
///
/// This keeps the latest BALs hot for repeated reads and drops old entries once the capacity is
/// reached.
#[derive(Debug, Clone)]
pub struct RecentBalStore {
    inner: Arc<RecentBalStoreInner>,
}

#[derive(Debug)]
struct RecentBalStoreInner {
    capacity: usize,
    entries: RwLock<HashMap<BlockHash, Bytes>>,
    block_index: RwLock<BTreeMap<BlockNumber, BlockHash>>,
}

impl RecentBalStore {
    /// Creates a new recent BAL store with the default capacity.
    pub fn new() -> Self {
        Self::with_capacity(DEFAULT_RECENT_BAL_CAPACITY)
    }

    /// Creates a new recent BAL store with the given capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            inner: Arc::new(RecentBalStoreInner {
                capacity,
                entries: RwLock::new(HashMap::new()),
                block_index: RwLock::new(BTreeMap::new()),
            }),
        }
    }
}

impl Default for RecentBalStore {
    fn default() -> Self {
        Self::new()
    }
}

impl BalStore for RecentBalStore {
    fn insert(
        &self,
        block_hash: BlockHash,
        block_number: BlockNumber,
        bal: Bytes,
    ) -> ProviderResult<()> {
        let mut entries = self.inner.entries.write();
        let mut block_index = self.inner.block_index.write();

        if let Some(old_hash) = block_index.get(&block_number) &&
            *old_hash != block_hash
        {
            entries.remove(old_hash);
        }

        if !entries.contains_key(&block_hash) &&
            entries.len() >= self.inner.capacity &&
            let Some((&oldest_number, &oldest_hash)) = block_index.first_key_value()
        {
            entries.remove(&oldest_hash);
            block_index.remove(&oldest_number);
        }

        entries.insert(block_hash, bal);
        block_index.insert(block_number, block_hash);

        Ok(())
    }

    fn get_by_hashes(&self, block_hashes: &[BlockHash]) -> ProviderResult<Vec<Option<Bytes>>> {
        let entries = self.inner.entries.read();
        Ok(block_hashes.iter().map(|hash| entries.get(hash).cloned()).collect())
    }

    fn get_by_range(&self, start: BlockNumber, count: u64) -> ProviderResult<Vec<Bytes>> {
        let entries = self.inner.entries.read();
        let block_index = self.inner.block_index.read();

        let mut result = Vec::new();
        for number in start..start.saturating_add(count) {
            let Some(hash) = block_index.get(&number) else { break };
            let Some(bal) = entries.get(hash) else { break };
            result.push(bal.clone());
        }

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloy_primitives::B256;

    #[test]
    fn insert_and_lookup_by_hash() {
        let store = RecentBalStore::with_capacity(8);
        let hash1 = B256::random();
        let hash2 = B256::random();
        let bal1 = Bytes::from_static(b"bal1");
        let bal2 = Bytes::from_static(b"bal2");

        store.insert(hash1, 1, bal1.clone()).unwrap();
        store.insert(hash2, 2, bal2.clone()).unwrap();

        let got = store.get_by_hashes(&[hash1, hash2, B256::random()]).unwrap();
        assert_eq!(got, vec![Some(bal1), Some(bal2), None]);
    }

    #[test]
    fn get_by_range_stops_at_gap() {
        let store = RecentBalStore::with_capacity(8);
        for number in [1, 2, 4, 5] {
            store
                .insert(B256::random(), number, Bytes::from(format!("bal{number}").into_bytes()))
                .unwrap();
        }

        assert_eq!(store.get_by_range(1, 5).unwrap().len(), 2);
        assert_eq!(store.get_by_range(4, 2).unwrap().len(), 2);
    }

    #[test]
    fn evicts_oldest_number_first() {
        let store = RecentBalStore::with_capacity(2);
        let hash1 = B256::random();
        let hash2 = B256::random();
        let hash3 = B256::random();

        store.insert(hash1, 10, Bytes::from_static(b"a")).unwrap();
        store.insert(hash2, 20, Bytes::from_static(b"b")).unwrap();
        store.insert(hash3, 30, Bytes::from_static(b"c")).unwrap();

        assert!(store.get_by_range(10, 1).unwrap().is_empty());
        assert_eq!(
            store.get_by_hashes(&[hash1, hash2, hash3]).unwrap(),
            vec![None, Some(Bytes::from_static(b"b")), Some(Bytes::from_static(b"c"))]
        );
        assert_eq!(store.get_by_range(20, 1).unwrap(), vec![Bytes::from_static(b"b")]);
    }

    #[test]
    fn replaces_reorged_entry_for_same_number() {
        let store = RecentBalStore::with_capacity(8);
        let old_hash = B256::random();
        let new_hash = B256::random();

        store.insert(old_hash, 7, Bytes::from_static(b"old")).unwrap();
        store.insert(new_hash, 7, Bytes::from_static(b"new")).unwrap();

        let by_hash = store.get_by_hashes(&[old_hash, new_hash]).unwrap();
        assert_eq!(by_hash, vec![None, Some(Bytes::from_static(b"new"))]);
        assert_eq!(store.get_by_range(7, 1).unwrap(), vec![Bytes::from_static(b"new")]);
    }
}
