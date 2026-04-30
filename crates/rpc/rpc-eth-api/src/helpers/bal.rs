//! Helpers for block access lists.
use alloy_consensus::BlockHeader;
use alloy_eip7928::{bal::Bal as AlloyBal, BlockAccessList};
use alloy_primitives::{BlockNumber, Bytes, B256};
use alloy_rpc_types_eth::BlockId;
use reth_errors::RethError;
use reth_evm::{block::BlockExecutor, ConfigureEvm, Evm};
use reth_primitives_traits::{BlockTy, Recovered, SealedBlock};
use reth_revm::{database::StateProviderDatabase, State};
use reth_rpc_eth_types::{
    cache::db::StateProviderTraitObjWrapper, error::FromEthApiError, EthApiError, StateCacheDb,
};
use reth_storage_api::{BalProvider, ProviderTx, StateProviderFactory};
use revm::state::bal::Bal;
use std::sync::Arc;
use tracing::debug;

use crate::{
    helpers::{Call, LoadBlock, Trace},
    RpcNodeCore,
};

/// Helper trait for `eth_blockAccessList` RPC method.
pub trait GetBlockAccessList: Trace + Call + LoadBlock {
    /// Retrieves the block access list for a block identified by its hash.
    fn get_block_access_list(
        &self,
        block_id: BlockId,
    ) -> impl Future<Output = Result<Option<BlockAccessList>, Self::Error>> + Send {
        async move {
            let block = self
                .recovered_block(block_id)
                .await?
                .ok_or_else(|| EthApiError::HeaderNotFound(block_id))?;

            self.spawn_blocking_io(move |eth_api| {
                let block_hash = block.hash();
                let block_number = block.number();

                if let Some(bal) = load_alloy_block_bal(eth_api.provider(), block_hash) {
                    return Ok(Some(bal))
                }

                let state = eth_api
                    .provider()
                    .state_by_block_id(block.parent_hash().into())
                    .map_err(Self::Error::from_eth_err)?;

                let mut db = State::builder()
                    .with_database(StateProviderDatabase::new(StateProviderTraitObjWrapper(state)))
                    .with_bal_builder()
                    .build();

                let block_txs = block.transactions_recovered();
                let mut executor = RpcNodeCore::evm_config(&eth_api)
                    .executor_for_block(&mut db, block.sealed_block())
                    .map_err(RethError::other)
                    .map_err(Self::Error::from_eth_err)?;

                executor.apply_pre_execution_changes().map_err(Self::Error::from_eth_err)?;
                executor.evm_mut().db_mut().bump_bal_index();

                // Advance the BAL index after each transaction so writes are recorded at the
                // matching block access index.
                for block_tx in block_txs {
                    executor.execute_transaction(block_tx).map_err(Self::Error::from_eth_err)?;
                    executor.evm_mut().db_mut().bump_bal_index();
                }

                executor
                    .apply_post_execution_changes()
                    .map_err(|err| EthApiError::Internal(err.into()))?;

                let bal = db.take_built_alloy_bal();
                if let Some(ref bal) = bal {
                    store_block_bal(eth_api.provider(), block_hash, block_number, bal);
                }
                Ok(bal)
            })
            .await
        }
    }
}

/// Loads the block BAL into `db` when it is available.
pub fn attach_block_bal<Provider>(provider: &Provider, block_hash: B256, db: &mut StateCacheDb)
where
    Provider: BalProvider,
{
    if let Some(bal) = load_revm_block_bal(provider, block_hash) {
        db.set_bal(Some(bal));
    }
}

/// Fetches and decodes the block BAL into the RPC representation.
fn load_alloy_block_bal<Provider>(provider: &Provider, block_hash: B256) -> Option<BlockAccessList>
where
    Provider: BalProvider,
{
    decode_stored_alloy_bal(provider, block_hash).map(AlloyBal::into_inner)
}

/// Fetches and decodes the block BAL into the revm representation.
pub fn load_revm_block_bal<Provider>(provider: &Provider, block_hash: B256) -> Option<Arc<Bal>>
where
    Provider: BalProvider,
{
    decode_stored_alloy_bal(provider, block_hash).and_then(|bal| {
        match Bal::try_from(bal.into_inner()) {
            Ok(bal) => Some(Arc::new(bal)),
            Err(err) => {
                debug!(
                    target: "reth::rpc",
                    ?block_hash,
                    %err,
                    "Failed to convert block access list"
                );
                None
            }
        }
    })
}

fn decode_stored_alloy_bal<Provider>(provider: &Provider, block_hash: B256) -> Option<AlloyBal>
where
    Provider: BalProvider,
{
    let raw_bal = match provider.bal_store().get_by_hashes(&[block_hash]) {
        Ok(bals) => bals.into_iter().next().flatten()?,
        Err(err) => {
            debug!(
                target: "reth::rpc",
                ?block_hash,
                %err,
                "Failed to fetch block access list"
            );
            return None
        }
    };

    decode_raw_block_bal(block_hash, raw_bal.as_ref())
}

fn store_block_bal<Provider>(
    provider: &Provider,
    block_hash: B256,
    block_number: BlockNumber,
    bal: &BlockAccessList,
) where
    Provider: BalProvider,
{
    let raw_bal = Bytes::from(alloy_rlp::encode(AlloyBal::from(bal.clone())));
    if let Err(err) = provider.bal_store().insert(block_hash, block_number, raw_bal) {
        debug!(
            target: "reth::rpc",
            ?block_hash,
            %err,
            "Failed to store block access list"
        );
    }
}

fn decode_raw_block_bal(block_hash: B256, raw_bal: &[u8]) -> Option<AlloyBal> {
    match alloy_rlp::decode_exact::<AlloyBal>(raw_bal) {
        Ok(bal) => Some(bal),
        Err(err) => {
            debug!(
                target: "reth::rpc",
                ?block_hash,
                %err,
                "Failed to decode block access list"
            );
            None
        }
    }
}

/// Positions `db` at the state before the transaction at `target_tx_index`.
///
/// If a BAL is attached, this only sets the BAL index. Otherwise it executes and commits the
/// transactions preceding `target_tx_index`.
///
/// The caller must apply the block's pre-execution changes before invoking this helper.
pub fn prepare_state_before_transaction<'a, Api, I>(
    api: &Api,
    db: &mut StateCacheDb,
    block: &SealedBlock<BlockTy<Api::Primitives>>,
    transactions: I,
    target_tx_index: usize,
) -> Result<(), Api::Error>
where
    Api: Call,
    I: IntoIterator<Item = Recovered<&'a ProviderTx<Api::Provider>>>,
{
    if db.bal_state.bal.is_some() {
        db.set_bal_index(target_tx_index as u64 + 1);
        return Ok(())
    }

    let mut executor = api
        .evm_config()
        .executor_for_block(db, block)
        .map_err(RethError::other)
        .map_err(Api::Error::from_eth_err)?;
    for tx in transactions.into_iter().take(target_tx_index) {
        executor.execute_transaction(tx).map_err(Api::Error::from_eth_err)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloy_eip7928::AccountChanges;
    use alloy_primitives::Address;
    use parking_lot::RwLock;
    use reth_storage_api::{errors::provider::ProviderResult, BalStore, BalStoreHandle};
    use std::{collections::HashMap, sync::Arc};

    #[derive(Clone, Default)]
    struct TestBalStore {
        entries: Arc<RwLock<HashMap<B256, Bytes>>>,
    }

    impl BalStore for TestBalStore {
        fn insert(
            &self,
            block_hash: B256,
            _block_number: BlockNumber,
            bal: Bytes,
        ) -> ProviderResult<()> {
            self.entries.write().insert(block_hash, bal);
            Ok(())
        }

        fn get_by_hashes(&self, block_hashes: &[B256]) -> ProviderResult<Vec<Option<Bytes>>> {
            let entries = self.entries.read();
            Ok(block_hashes.iter().map(|hash| entries.get(hash).cloned()).collect())
        }

        fn get_by_range(&self, _start: BlockNumber, _count: u64) -> ProviderResult<Vec<Bytes>> {
            Ok(Vec::new())
        }
    }

    struct TestBalProvider {
        store: BalStoreHandle,
    }

    impl TestBalProvider {
        fn new() -> Self {
            Self { store: BalStoreHandle::new(TestBalStore::default()) }
        }
    }

    impl BalProvider for TestBalProvider {
        fn bal_store(&self) -> &BalStoreHandle {
            &self.store
        }
    }

    #[test]
    fn load_alloy_block_bal_reads_stored_bal() {
        let provider = TestBalProvider::new();
        let block_hash = B256::random();
        let bal = vec![AccountChanges::new(Address::from([0x11; 20]))];
        let raw_bal = Bytes::from(alloy_rlp::encode(AlloyBal::from(bal.clone())));

        provider.bal_store().insert(block_hash, 1, raw_bal).unwrap();

        assert_eq!(load_alloy_block_bal(&provider, block_hash), Some(bal));
    }

    #[test]
    fn load_alloy_block_bal_ignores_invalid_stored_bal() {
        let provider = TestBalProvider::new();
        let block_hash = B256::random();

        provider.bal_store().insert(block_hash, 1, Bytes::from_static(b"invalid")).unwrap();

        assert_eq!(load_alloy_block_bal(&provider, block_hash), None);
    }

    #[test]
    fn store_block_bal_writes_rlp_bal() {
        let provider = TestBalProvider::new();
        let block_hash = B256::random();
        let block_number = 7;
        let bal = vec![AccountChanges::new(Address::from([0x22; 20]))];

        store_block_bal(&provider, block_hash, block_number, &bal);

        let stored = provider.bal_store().get_by_hashes(&[block_hash]).unwrap().remove(0).unwrap();
        assert_eq!(
            decode_raw_block_bal(block_hash, stored.as_ref()).map(AlloyBal::into_inner),
            Some(bal)
        );
    }
}
