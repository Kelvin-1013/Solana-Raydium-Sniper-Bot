# Solana Raydium Sniper Bot  [📞👉](https://t.me/blockchainDeveloper_Ben)

## Overview

This project implements a Solana Raydium sniper bot, designed to monitor and interact with Raydium pools on the Solana blockchain. The bot can fetch pool prices, execute swaps, and automatically trade based on predefined conditions.
(Main project is in Private repo)


## Features

- Fetch pool state and price information
- Execute swaps on Raydium pools
- Monitor pool prices and execute trades when target prices are reached
- Support for both buying and selling operations
- Customizable slippage and swap amount settings
- Integration with Jito (optional)

## Prerequisites

- Rust programming environment
- Solana CLI tools
- Access to a Solana RPC endpoint

## Configuration

The bot uses environment variables for configuration. Create a `.env` file in the project root with the following variables:

```
RPC_URL=<Your Solana RPC URL>
PRIVATE_KEY=<Your wallet private key>
TARGET_ADDRESS=<Target pool address>
TARGET_PRICE=<Target price for trading>
MINT_ADDRESS=<Token mint address>
UNIT_PRICE=<Compute unit price (optional)>
UNIT_LIMIT=<Compute unit limit (optional)>
TX_SIMULATE=<true/false for transaction simulation>
HTTP_PROXY=<HTTP proxy URL (optional)>
```

## Main Components

1. **Pool State and Price Fetching**: 
   - `get_pool_state`: Retrieves the state of a Raydium pool
   - `get_pool_price`: Calculates and returns the current price of a pool

2. **Swap Execution**:
   - `swap`: Main function for executing swaps on Raydium pools
   - `amm_swap`: Builds the swap instruction

3. **Account and Mint Information**:
   - `get_account_info`: Retrieves token account information
   - `get_mint_info`: Fetches mint information for a token

4. **Transaction Handling**:
   - `new_signed_and_send`: Creates and sends transactions

5. **Main Loop**:
   - Monitors pool price
   - Executes swaps when the target price is reached

## Usage

1. Set up the environment variables in the `.env` file.
2. Run the bot using `cargo run`.

The bot will continuously monitor the specified pool's price and execute a swap when the target price is reached.

## Error Handling and Logging

The bot includes comprehensive error handling and logging to help diagnose issues and track its operation. Key information and errors are logged using the `tracing` crate.

## Security Considerations

- Keep your private key secure and never share it.
- Use a dedicated wallet for the bot to limit potential losses.
- Test thoroughly with small amounts before deploying with significant funds.

