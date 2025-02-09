use solana_client::rpc_client::RpcClient;
use solana_client::nonblocking::rpc_client::RpcClient as NonblockingRpcClient;
use std::str::FromStr;
use std::sync::Arc;
use raydium_amm::state::{Loadable, AmmInfo};
use anyhow::{anyhow, Context, Result};
use common::common_utils;
use spl_token_2022::{
    amount_to_ui_amount,
    extension::StateWithExtensionsOwned,
    state::{Account, Mint as TokenMint},
};
use tracing::debug;
use solana_client::rpc_filter::{Memcmp, RpcFilterType};
use serde::Deserialize;
use std::env;
use reqwest::Proxy;
use solana_sdk::{
    signer::{keypair::Keypair, Signer},
    program_pack::Pack, 
    pubkey::Pubkey,
    system_instruction,
    instruction::Instruction,
    transaction::Transaction,
    signature::Signature,
};
use clap::ValueEnum;
use spl_associated_token_account::{
    get_associated_token_address, 
    instruction::create_associated_token_account
};
use spl_token_client::{
    client::{ProgramClient, ProgramRpcClient, ProgramRpcClientSendTransaction},
    token::{TokenError, TokenResult},
};
use tracing::{info, warn, error};
use spl_token::ui_amount_to_amount;
use amm_cli::AmmSwapInfoResult;
use dotenv::dotenv;

#[derive(Debug, Deserialize, Clone)]
pub struct Pool {
    pub id: String,
    #[serde(rename = "programId")]
    pub program_id: String,
    #[serde(rename = "mintA")]
    pub mint_a: PoolMint,
    #[serde(rename = "mintB")]
    pub mint_b: PoolMint,
    #[serde(rename = "marketId")]
    pub market_id: String,
}

#[derive(Debug, Deserialize, Clone)]
pub struct PoolMint {
    pub address: String,
    pub symbol: String,
    pub name: String,
    pub decimals: u8,
}


#[derive(Debug, Deserialize)]
pub struct PoolData {
    pub data: Vec<Pool>,
}

#[derive(ValueEnum, Debug, Clone, Deserialize)]
pub enum SwapDirection {
    #[serde(rename = "buy")]
    Buy,
    #[serde(rename = "sell")]
    Sell,
}
impl From<SwapDirection> for u8 {
    fn from(value: SwapDirection) -> Self {
        match value {
            SwapDirection::Buy => 0,
            SwapDirection::Sell => 1,
        }
    }
}

impl PoolData {
    pub fn get_pool(&self) -> Option<Pool> {
        self.data.first().cloned()
    }
}

#[derive(Debug, Deserialize)]
pub struct PoolInfo {
    pub success: bool,
    pub data: PoolData,
}

#[derive(ValueEnum, Debug, Clone, Deserialize)]
pub enum SwapInType {
    /// Quantity
    #[serde(rename = "qty")]
    Qty,
    /// Percentage
    #[serde(rename = "pct")]
    Pct,
}

pub fn get_wallet() -> Result<Arc<Keypair>> {
    let wallet = Keypair::from_base58_string(&env::var("PRIVATE_KEY")?);
    return Ok(Arc::new(wallet));
}

pub const AMM_PROGRAM: &str = "675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8";

async fn get_pool_state_by_mint(
    rpc_client: Arc<solana_client::rpc_client::RpcClient>,
    mint: &str,
) -> Result<(Pubkey, AmmInfo)> {
    debug!("finding pool state by mint: {}", mint);
    
    // Define the expected size of AmmInfo
    const AMM_INFO_SIZE: usize = 752;  // Fixed size for Raydium AMM v4
    debug!("Expected AmmInfo size: {}", AMM_INFO_SIZE);
    debug!("AmmInfo struct size: {}", std::mem::size_of::<AmmInfo>());
    
    // (pc_mint, coin_mint)
    let pairs = vec![
        // pump pool
        (
            Some(spl_token::native_mint::ID),
            Pubkey::from_str(mint).ok(),
        ),
        // general pool
        (
            Pubkey::from_str(mint).ok(),
            Some(spl_token::native_mint::ID),
        ),
    ];

    let amm_program = Pubkey::from_str(AMM_PROGRAM)?;
    // Find matching AMM pool from mint pairs by filter
    let mut found_pools = None;
    for (coin_mint, pc_mint) in pairs {
        debug!(
            "get_pool_state_by_mint filter: coin_mint: {:?}, pc_mint: {:?}",
            coin_mint, pc_mint
        );
        let filters = match (coin_mint, pc_mint) {
            (None, None) => Some(vec![RpcFilterType::DataSize(AMM_INFO_SIZE as u64)]),
            (Some(coin_mint), None) => Some(vec![
                RpcFilterType::Memcmp(Memcmp::new_base58_encoded(400, &coin_mint.to_bytes())),
                RpcFilterType::DataSize(AMM_INFO_SIZE as u64),
            ]),
            (None, Some(pc_mint)) => Some(vec![
                RpcFilterType::Memcmp(Memcmp::new_base58_encoded(432, &pc_mint.to_bytes())),
                RpcFilterType::DataSize(AMM_INFO_SIZE as u64),
            ]),
            (Some(coin_mint), Some(pc_mint)) => Some(vec![
                RpcFilterType::Memcmp(Memcmp::new_base58_encoded(400, &coin_mint.to_bytes())),
                RpcFilterType::Memcmp(Memcmp::new_base58_encoded(432, &pc_mint.to_bytes())),
                RpcFilterType::DataSize(AMM_INFO_SIZE as u64),
            ]),
        };
        let pools =
            common::rpc::get_program_accounts_with_filters(&rpc_client, amm_program, filters)
                .unwrap();
        if !pools.is_empty() {
            found_pools = Some(pools);
            break;
        }
    }

    match found_pools {
        Some(pools) => {
            let pool = &pools[0];
            debug!("Found pool with ID: {}", pool.0);
            debug!("Actual account data size: {}", pool.1.data.len());
            debug!("First few bytes of data: {:?}", &pool.1.data[..std::cmp::min(32, pool.1.data.len())]);
            
            // Ensure the data length matches expected size
            if pool.1.data.len() != AMM_INFO_SIZE {
                return Err(anyhow!(
                    "Invalid data size: expected {} but got {}",
                    AMM_INFO_SIZE,
                    pool.1.data.len()
                ));
            }
            
            // Try loading with detailed error handling
            match raydium_amm::state::AmmInfo::load_from_bytes(&pool.1.data) {
                Ok(pool_state) => {
                    debug!("Successfully loaded AmmInfo");
                    Ok((pool.0, pool_state.clone()))
                },
                Err(e) => {
                    error!("Failed to load AmmInfo: {:?}", e);
                    error!("Data length: {}", pool.1.data.len());
                    Err(anyhow!("Failed to load AMM info: {:?}", e))
                }
            }
        }
        None => {
            error!("No pools found for mint: {}", mint);
            Err(anyhow!("NotFoundPool: pool state not found"))
        }
    }
}

async fn get_pool_info(mint1: &str, mint2: &str) -> Result<PoolData> {
    let mut client_builder = reqwest::Client::builder();
    if let Ok(http_proxy) = env::var("HTTP_PROXY") {
        let proxy = Proxy::all(http_proxy)?;
        client_builder = client_builder.proxy(proxy);
    }
    let client = client_builder.build()?;

    let result = client
        .get("https://api-v3.raydium.io/pools/info/mint")
        .query(&[
            ("mint1", mint1),
            ("mint2", mint2),
            ("poolType", "standard"),
            ("poolSortField", "default"),
            ("sortType", "desc"),
            ("pageSize", "1"),
            ("page", "1"),
        ])
        .send()
        .await?
        .json::<PoolInfo>()
        .await
        .context("Failed to parse pool info JSON")?;
    Ok(result.data)
}

async fn get_pool_state(
    rpc_client: Arc<solana_client::rpc_client::RpcClient>,
    pool_id: Option<&str>,
    mint: Option<&str>,
) -> Result<(Pubkey, AmmInfo)> {
    if let Some(pool_id) = pool_id {
        debug!("finding pool state by pool_id: {}", pool_id);
        let amm_pool_id = Pubkey::from_str(pool_id)?;
        let account_data = common::rpc::get_account(&rpc_client, &amm_pool_id)?
            .ok_or(anyhow!("NotFoundPool: pool state not found"))?;
        
        // Check if we're dealing with a v4 or v3 pool
        let pool_state = if account_data.len() == 752 {
            // V4 pool
            AmmInfo::load_from_bytes(&account_data)?.to_owned()
        } else if account_data.len() == 637 {
            // V3 pool
            let mut padded_data = vec![0u8; 752];
            padded_data[..account_data.len()].copy_from_slice(&account_data);
            AmmInfo::load_from_bytes(&padded_data)?.to_owned()
        } else {
            return Err(anyhow!(
                "Unexpected account data size: {}. Expected either 752 (v4) or 637 (v3)",
                account_data.len()
            ));
        };

        Ok((amm_pool_id, pool_state))
    } else {
        println!("Debugging ... false");
        if let Some(mint) = mint {
            // Try both methods with better error handling
            match get_pool_state_by_mint(rpc_client.clone(), mint).await {
                Ok(result) => Ok(result),
                Err(e) => {
                    debug!("Failed to get pool by mint via RPC: {:?}", e);
                    // Try via Raydium API as fallback
                    match get_pool_info(&spl_token::native_mint::ID.to_string(), mint).await {
                        Ok(pool_data) => {
                            match pool_data.get_pool() {
                                Some(pool) => {
                                    let amm_pool_id = Pubkey::from_str(&pool.id)?;
                                    debug!("Found pool via Raydium API: {}", amm_pool_id);
                                    let account_data = common::rpc::get_account(&rpc_client, &amm_pool_id)?
                                        .ok_or(anyhow!("NotFoundPool: pool state not found"))?;
                                    
                                    // Apply the same version check here
                                    let pool_state = if account_data.len() == 752 {
                                        AmmInfo::load_from_bytes(&account_data)?.to_owned()
                                    } else if account_data.len() == 637 {
                                        let mut padded_data = vec![0u8; 752];
                                        padded_data[..account_data.len()].copy_from_slice(&account_data);
                                        let state = AmmInfo::load_from_bytes(&padded_data)?.to_owned();
                                        state
                                    } else {
                                        return Err(anyhow!(
                                            "Unexpected account data size: {}. Expected either 752 (v4) or 637 (v3)",
                                            account_data.len()
                                        ));
                                    };
                                    
                                    Ok((amm_pool_id, pool_state))
                                },
                                None => Err(anyhow!("NotFoundPool: pool not found in raydium api"))
                            }
                        },
                        Err(e) => Err(anyhow!("Failed to get pool info from Raydium API: {:?}", e))
                    }
                }
            }
        } else {
            Err(anyhow!("NotFoundPool: pool state not found"))
        }
    }
}

async fn get_pool_price(
  pool_id: Option<&str>,
  mint: Option<&str>,
) -> Result<(f64, f64, f64)> {
    println!("Get Pool Price ...");
    let rpc_url = env::var("RPC_URL").expect("RPC_URL environment variable not set");
    let rpc_client = RpcClient::new(rpc_url);
    let client = Arc::new(rpc_client);

  let (amm_pool_id, pool_state) = get_pool_state(client.clone(), pool_id, mint).await?;
  
//   println!("pool_state : {:#?}", pool_state);
  
  let load_pubkeys = vec![pool_state.pc_vault, pool_state.coin_vault];
  let rsps = common::rpc::get_multiple_accounts(&client, &load_pubkeys).unwrap();
  
  // Add proper error handling for vault accounts
  let amm_pc_vault_account = rsps[0].clone()
      .ok_or_else(|| anyhow!("Failed to fetch PC vault account"))?;
  let amm_coin_vault_account = rsps[1].clone()
      .ok_or_else(|| anyhow!("Failed to fetch coin vault account"))?;
  
  let amm_pc_vault = common_utils::unpack_token(&amm_pc_vault_account.data)
      .map_err(|e| anyhow!("Failed to unpack PC vault token: {}", e))?;
  let amm_coin_vault = common_utils::unpack_token(&amm_coin_vault_account.data)
      .map_err(|e| anyhow!("Failed to unpack coin vault token: {}", e))?;
  
  let (base_account, quote_account) = if amm_coin_vault.base.is_native() {
      (
          (
              pool_state.pc_vault_mint,
              amount_to_ui_amount(amm_pc_vault.base.amount, pool_state.pc_decimals as u8),
          ),
          (
              pool_state.coin_vault_mint,
              amount_to_ui_amount(amm_coin_vault.base.amount, pool_state.coin_decimals as u8),
          ),
      )
  } else {
      (
          (
              pool_state.coin_vault_mint,
              amount_to_ui_amount(amm_coin_vault.base.amount, pool_state.coin_decimals as u8),
          ),
          (
              pool_state.pc_vault_mint,
              amount_to_ui_amount(amm_pc_vault.base.amount, pool_state.pc_decimals as u8),
          ),
      )
  };

  let price = quote_account.1 / base_account.1;
  
  println!(
      "calculate pool[{}]: {}: {}, {}: {}, price: {} sol",
      amm_pool_id, base_account.0, base_account.1, quote_account.0, quote_account.1, price
  );

  Ok((base_account.1, quote_account.1, price))
}

async fn get_account_info(
    client: Arc<NonblockingRpcClient>,
    _keypair: Arc<Keypair>,
    address: &Pubkey,
    account: &Pubkey,
) -> TokenResult<StateWithExtensionsOwned<Account>> {
    let program_client = Arc::new(ProgramRpcClient::new(
        client.clone(),
        ProgramRpcClientSendTransaction,
    ));
    let account = program_client
        .get_account(*account)
        .await
        .map_err(TokenError::Client)?
        .ok_or(TokenError::AccountNotFound)
        .inspect_err(|err| warn!("{} {}: mint {}", account, err, address))?;

    if account.owner != spl_token::ID {
        return Err(TokenError::AccountInvalidOwner);
    }
    let account = StateWithExtensionsOwned::<Account>::unpack(account.data)?;
    if account.base.mint != *address {
        return Err(TokenError::AccountInvalidMint);
    }

    Ok(account)
}


fn get_unit_limit() -> u32 {
    env::var("UNIT_LIMIT")
        .ok()
        .and_then(|v| u32::from_str(&v).ok())
        .unwrap_or(200_000)
}

async fn new_signed_and_send(
    client: &RpcClient,
    keypair: &Keypair,
    mut instructions: Vec<Instruction>,
    use_jito: bool,
) -> Result<Vec<String>> {
    let unit_limit = get_unit_limit();
    let unit_price = get_unit_price();
    // If not using Jito, manually set the compute unit price and limit
    if !use_jito {
        let modify_compute_units =
            solana_sdk::compute_budget::ComputeBudgetInstruction::set_compute_unit_limit(
                unit_limit,
            );
        let add_priority_fee =
            solana_sdk::compute_budget::ComputeBudgetInstruction::set_compute_unit_price(
                unit_price,
            );
        instructions.insert(0, modify_compute_units);
        instructions.insert(1, add_priority_fee);
    }
    // send init tx
    let recent_blockhash = client.get_latest_blockhash()?;
    let txn = Transaction::new_signed_with_payer(
        &instructions,
        Some(&keypair.pubkey()),
        &vec![&*keypair],
        recent_blockhash,
    );

    if env::var("TX_SIMULATE").ok() == Some("true".to_string()) {
        let simulate_result = client.simulate_transaction(&txn)?;
        if let Some(logs) = simulate_result.value.logs {
            for log in logs {
                info!("{}", log);
            }
        }
        return match simulate_result.value.err {
            Some(err) => Err(anyhow!("{}", err)),
            None => Ok(vec![]),
        };
    }

    if use_jito {
        // jito implementation placeholder
        Ok(vec![])
    } else {
        let sig = common::rpc::send_txn(&client, &txn, true)?;
        info!("signature: {:?}", sig);
        Ok(vec![sig.to_string()])
    }
}


#[tokio::main]
async fn main() -> Result<()> {
    dotenv().ok();
    let pool_id = env::var("TARGET_ADDRESS").context("TARGET_ADDRESS environment variable not set")?;
    let target_price = env::var("TARGET_PRICE")
        .context("TARGET_PRICE environment variable not set")?
        .parse::<f64>()
        .context("Failed to parse TARGET_PRICE as f64")?;
    
    let swap_amount = 1.0;  // 100% of tokens
    let rpc_url = env::var("RPC_URL").expect("RPC_URL environment variable not set");
    let rpc_client = Arc::new(RpcClient::new(rpc_url));

    let wallet = get_wallet()?;
    let mint = env::var("MINT_ADDRESS")?;

    loop {
        match swap(
            Some(&pool_id),
            wallet.clone(),
            &mint,
            swap_amount,
            SwapDirection::Sell,
            SwapInType::Pct,
            10,
            false
        ).await {
            Ok(signatures) => {
                println!("Swap initiated. Waiting for confirmation...");
                
                // Wait for each transaction to confirm
                for signature in signatures {
                    let sig = Signature::from_str(&signature)?;
                    if rpc_client.confirm_transaction(&sig)? {
                        match rpc_client.get_signature_status(&sig) {
                            Ok(status) => {
                                if status.is_none() {
                                    println!("Swap transaction {} confirmed successfully!", signature);
                                } else {
                                    error!("Swap transaction {} failed with error: {:?}", signature, status);
                                    continue;
                                }
                            }
                            _ => {
                                error!("Failed to get transaction status for {}", signature);
                                continue;
                            }
                        }
                    } else {
                        error!("Failed to confirm transaction {}", signature);
                        continue;
                    }
                }
                
                // Optional: Verify the token balance is now 0
                let token_ata = get_associated_token_address(&wallet.pubkey(), &Pubkey::from_str(&mint)?);
                match rpc_client.get_token_account_balance(&token_ata) {
                    Ok(balance) => {
                        if balance.amount == "0" {
                            println!("Swap completed successfully! Token balance is now 0");
                            break;  // Exit the monitoring loop
                        } else {
                            println!("Warning: Token balance is not 0 after swap: {}", balance.amount);
                        }
                    }
                    Err(e) => {
                        error!("Failed to check final token balance: {}", e);
                    }
                }
            }
            Err(e) => {
                error!("Failed to initiate swap: {}", e);
            }
        }
        match get_pool_price(Some(&pool_id), None).await {
            Ok((_base_amount, _quote_amount, current_price)) => {
                println!("Current price: {} SOL", current_price);
                
                if current_price > target_price {
                    println!("Price threshold reached! Initiating swap of all tokens to SOL...");
                    
                }
            }
            Err(e) => eprintln!("Error fetching pool price: {}", e),
        }

        tokio::time::sleep(tokio::time::Duration::from_secs(5)).await;
    }
    
    Ok(())
}
