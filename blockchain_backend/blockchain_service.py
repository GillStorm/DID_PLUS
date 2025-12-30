"""
Blockchain Service Module for DID++ Project
Implements Web3.py integration with Sepolia testnet for DID registration,
verification logging, and history querying.  

Author: Backend Engineer
Date: 2025-12-30
"""

import os
import json
import hashlib
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
from dotenv import load_dotenv
from web3 import Web3
from web3.contract import Contract

# Load environment variables
load_dotenv()


@dataclass
class VerificationRecord:
    """Data class for verification records from blockchain"""
    verification_hash: str
    verified:  bool
    timestamp: int
    
    def to_dict(self) -> Dict:
        """Convert to dictionary format"""
        return {
            "verification_hash": self.verification_hash,
            "verified": self.verified,
            "timestamp":  self.timestamp,
            "timestamp_readable": datetime.fromtimestamp(self. timestamp).isoformat()
        }


@dataclass
class IdentityRecord:
    """Data class for identity records from blockchain"""
    identity_hash: str
    created_at: int
    exists: bool
    
    def to_dict(self) -> Dict:
        """Convert to dictionary format"""
        return {
            "identity_hash": self.identity_hash,
            "created_at":  self.created_at,
            "created_at_readable": datetime.fromtimestamp(self.created_at).isoformat() if self.created_at else None,
            "exists": self.exists
        }


class BlockchainService:
    """
    Service class for interacting with DID Registry smart contract on Sepolia testnet. 
    
    Environment Variables Required:
    - SEPOLIA_RPC_URL: Alchemy Sepolia RPC endpoint
    - BACKEND_PRIVATE_KEY: Private key for transaction signing
    - CONTRACT_ADDRESS:  Deployed DID Registry contract address
    """
    
    def __init__(self):
        """Initialize Blockchain Service with Web3 connection and contract instance"""
        # Load environment variables
        self.rpc_url = os.getenv("SEPOLIA_RPC_URL")
        self.private_key = os.getenv("BACKEND_PRIVATE_KEY")
        self.contract_address = os.getenv("CONTRACT_ADDRESS")
        
        # Validate environment variables
        self._validate_env_variables()
        
        # Initialize Web3 connection
        self.w3 = Web3(Web3.HTTPProvider(self.rpc_url))
        
        # Validate connection
        if not self.w3.is_connected():
            raise ConnectionError(f"Failed to connect to Sepolia RPC at {self.rpc_url}")
        
        # Load contract ABI
        self.contract_abi = self._load_contract_abi()
        
        # Initialize contract instance
        self.contract: Contract = self. w3.eth.contract(
            address=Web3.to_checksum_address(self. contract_address),
            abi=self.contract_abi
        )
        
        # Derive account from private key
        self.account = self. w3.eth.account.from_key(self.private_key)
        self.account_address = self.account.address
        
        # Get gas price for transaction estimation
        self.base_gas_price = self.w3.eth.gas_price
    
    @staticmethod
    def _validate_env_variables() -> None:
        """Validate required environment variables are set"""
        required_vars = ["SEPOLIA_RPC_URL", "BACKEND_PRIVATE_KEY", "CONTRACT_ADDRESS"]
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        
        if missing_vars:
            raise ValueError(
                f"Missing required environment variables:  {', '.join(missing_vars)}"
            )
    
    @staticmethod
    def _load_contract_abi() -> List:
        """Load contract ABI from contract_abi.json"""
        abi_path = os.path.join(os.path.dirname(__file__), "contract_abi.json")
        
        try:
            with open(abi_path, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Contract ABI file not found at {abi_path}")
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON in contract ABI file: {abi_path}")
    
    @staticmethod
    def calculate_identity_hash(identity_data: str) -> bytes:
        """
        Calculate SHA256 hash of identity data (Stage 3 of pipeline).
        
        Args:
            identity_data:  String representation of identity information
                          (e.g., JSON string of multimodal verification scores)
        
        Returns: 
            bytes: 32-byte SHA256 hash
        """
        if not isinstance(identity_data, str):
            raise TypeError("identity_data must be a string")
        
        return hashlib.sha256(identity_data.encode()).digest()
    
    def register_did(
        self,
        did: str,
        identity_data: str,
        gas_limit: int = 150000
    ) -> Dict:
        """
        Register a DID on the blockchain (Flow 1).
        
        Args:
            did: Decentralized Identifier string
            identity_data: String representation of identity information for hashing
            gas_limit: Gas limit for transaction (default: 150,000)
        
        Returns: 
            Dict containing: 
            - tx_hash: Transaction hash
            - tx_receipt: Transaction receipt
            - block_number: Block number where transaction was included
            - status: Transaction status (1 = success)
        
        Raises:
            ValueError: If DID is empty or identity_data cannot be hashed
            TransactionFailed: If transaction fails
        """
        # Validate inputs
        if not did or not isinstance(did, str):
            raise ValueError("DID must be a non-empty string")
        
        if not identity_data or not isinstance(identity_data, str):
            raise ValueError("Identity data must be a non-empty string")
        
        # Calculate identity hash
        identity_hash = self.calculate_identity_hash(identity_data)
        
        try:
            # Build transaction
            tx_dict = self.contract.functions.registerDID(
                did,
                identity_hash
            ).build_transaction({
                "from": self.account_address,
                "nonce": self. w3.eth.get_transaction_count(self.account_address),
                "gasPrice": self.base_gas_price,
                "gas": gas_limit,
                "chainId": 11155111,  # Sepolia Chain ID
            })
            
            # Sign transaction
            signed_tx = self.w3.eth. account.sign_transaction(tx_dict, self.private_key)
            
            # Send transaction
            tx_hash = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)
            
            # Wait for confirmation
            tx_receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash, timeout=300)
            
            # Validate transaction success
            if tx_receipt["status"] != 1:
                raise Exception(
                    f"Registration transaction failed for DID: {did}. "
                    f"Tx Hash: {tx_hash. hex()}"
                )
            
            return {
                "tx_hash":  tx_hash. hex(),
                "tx_receipt": dict(tx_receipt),
                "block_number": tx_receipt["blockNumber"],
                "status":  "success",
                "gas_used": tx_receipt["gasUsed"],
                "did": did,
                "identity_hash": identity_hash. hex()
            }
        
        except Exception as e: 
            raise Exception(f"Failed to register DID {did}: {str(e)}")
    
    def log_verification(
        self,
        verification_hash: bytes,
        did: str,
        verified:  bool,
        final_score: float = None,
        verification_threshold: float = 0.7,
        gas_limit: int = 120000
    ) -> Dict:
        """
        Log verification result on blockchain (Flow 2).
        
        Args:
            verification_hash: 32-byte hash of verification process
            did:  Decentralized Identifier
            verified: Boolean result of verification (only logs if True)
            final_score: Final verification score (optional, for logging purposes)
            verification_threshold:  Threshold for verification (default: 0.7)
            gas_limit:  Gas limit for transaction (default: 120,000)
        
        Returns:
            Dict containing: 
            - tx_hash: Transaction hash
            - status: Transaction status
            - verified:  Verification result logged
            - block_number: Block where transaction was included
        
        Raises:
            ValueError: If verification_hash is invalid or score below threshold
            Exception: If transaction fails
        """
        # Validate inputs
        if not isinstance(verification_hash, bytes) or len(verification_hash) != 32:
            raise ValueError("Verification hash must be 32 bytes")
        
        if not did or not isinstance(did, str):
            raise ValueError("DID must be a non-empty string")
        
        # Check verification threshold
        if final_score is not None and final_score < verification_threshold:
            verified = False
        
        try:
            # Build transaction
            tx_dict = self.contract. functions.logVerification(
                verification_hash,
                did,
                verified
            ).build_transaction({
                "from": self.account_address,
                "nonce": self.w3.eth.get_transaction_count(self.account_address),
                "gasPrice":  self.base_gas_price,
                "gas": gas_limit,
                "chainId": 11155111,  # Sepolia Chain ID
            })
            
            # Sign transaction
            signed_tx = self.w3.eth.account.sign_transaction(tx_dict, self.private_key)
            
            # Send transaction
            tx_hash = self.w3.eth.send_raw_transaction(signed_tx. rawTransaction)
            
            # Wait for confirmation
            tx_receipt = self.w3.eth. wait_for_transaction_receipt(tx_hash, timeout=300)
            
            # Validate transaction success
            if tx_receipt["status"] != 1:
                raise Exception(
                    f"Verification logging transaction failed for DID: {did}. "
                    f"Tx Hash: {tx_hash.hex()}"
                )
            
            return {
                "tx_hash": tx_hash. hex(),
                "status": "success",
                "verified": verified,
                "block_number": tx_receipt["blockNumber"],
                "gas_used": tx_receipt["gasUsed"],
                "did": did,
                "final_score": final_score,
                "threshold": verification_threshold
            }
        
        except Exception as e:
            raise Exception(f"Failed to log verification for DID {did}: {str(e)}")
    
    def get_on_chain_history(self, did: str) -> List[VerificationRecord]: 
        """
        Retrieve verification history from blockchain (Flow 3).
        
        Args:
            did:  Decentralized Identifier
        
        Returns:
            List of VerificationRecord objects containing:
            - verification_hash:  Hash of verification
            - verified: Boolean verification result
            - timestamp: Block timestamp when logged
        
        Raises:
            ValueError: If DID is invalid
            Exception: If contract call fails
        """
        # Validate input
        if not did or not isinstance(did, str):
            raise ValueError("DID must be a non-empty string")
        
        try:
            # Call contract function (read-only, no gas cost)
            verifications = self.contract.functions. getVerifications(did).call()
            
            # Convert to VerificationRecord objects
            records = [
                VerificationRecord(
                    verification_hash=f"0x{verification[0]. hex()}",
                    verified=verification[1],
                    timestamp=verification[2]
                )
                for verification in verifications
            ]
            
            return records
        
        except Exception as e:
            raise Exception(f"Failed to get verification history for DID {did}: {str(e)}")
    
    def get_identity_record(self, did: str) -> IdentityRecord:
        """
        Retrieve identity record from blockchain. 
        
        Args:
            did: Decentralized Identifier
        
        Returns:
            IdentityRecord containing identity hash, creation time, and existence status
        
        Raises: 
            ValueError: If DID is invalid
        """
        # Validate input
        if not did or not isinstance(did, str):
            raise ValueError("DID must be a non-empty string")
        
        try:
            # Call contract function
            identity = self.contract.functions.getIdentity(did).call()
            
            return IdentityRecord(
                identity_hash=f"0x{identity[0]. hex()}",
                created_at=identity[1],
                exists=identity[2]
            )
        
        except Exception as e:
            raise Exception(f"Failed to get identity record for DID {did}: {str(e)}")
    
    def did_exists(self, did: str) -> bool:
        """
        Check if a DID is already registered on the blockchain.
        
        Args:
            did:  Decentralized Identifier
        
        Returns:
            bool: True if DID exists, False otherwise
        """
        # Validate input
        if not did or not isinstance(did, str):
            raise ValueError("DID must be a non-empty string")
        
        try:
            return self.contract.functions.didExists(did).call()
        except Exception as e:
            raise Exception(f"Failed to check if DID exists {did}: {str(e)}")
    
    def get_verification_count(self, did: str) -> int:
        """
        Get the total number of verifications logged for a DID.
        
        Args:
            did:  Decentralized Identifier
        
        Returns:
            int: Number of verifications
        """
        # Validate input
        if not did or not isinstance(did, str):
            raise ValueError("DID must be a non-empty string")
        
        try:
            return self.contract.functions.getVerificationCount(did).call()
        except Exception as e: 
            raise Exception(f"Failed to get verification count for DID {did}: {str(e)}")
    
    def estimate_registration_gas(self, did: str, identity_data: str) -> int:
        """
        Estimate gas cost for DID registration.
        
        Args:
            did: Decentralized Identifier
            identity_data: String representation of identity information
        
        Returns:
            int:  Estimated gas cost
        """
        try:
            identity_hash = self.calculate_identity_hash(identity_data)
            
            return self.contract.functions.registerDID(
                did,
                identity_hash
            ).estimate_gas({"from": self.account_address})
        except Exception as e: 
            raise Exception(f"Failed to estimate registration gas for DID {did}: {str(e)}")
    
    def estimate_verification_gas(
        self,
        verification_hash: bytes,
        did: str,
        verified: bool
    ) -> int:
        """
        Estimate gas cost for verification logging.
        
        Args:
            verification_hash: 32-byte hash of verification
            did:  Decentralized Identifier
            verified: Boolean verification result
        
        Returns: 
            int: Estimated gas cost
        """
        try:
            return self.contract.functions.logVerification(
                verification_hash,
                did,
                verified
            ).estimate_gas({"from": self.account_address})
        except Exception as e: 
            raise Exception(f"Failed to estimate verification gas for DID {did}: {str(e)}")