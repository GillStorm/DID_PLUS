"""
Unit and Integration Tests for Blockchain Service

Run tests with:
    pytest test_blockchain_service.py -v

Run with coverage:
    pytest test_blockchain_service.py --cov=blockchain_service
"""

import json
import pytest
import hashlib
from unittest.mock import Mock, patch, MagicMock
from blockchain_service import (
    BlockchainService,
    VerificationRecord,
    IdentityRecord
)


# ============================================================================
# Test Data Setup
# ============================================================================

TEST_DID = "did:example:test123"
TEST_IDENTITY_DATA = json.dumps({
    "face_score": 0.92,
    "voice_score": 0.88,
    "document_score": 0.95,
    "did":  TEST_DID
}, sort_keys=True)

TEST_IDENTITY_HASH = hashlib.sha256(TEST_IDENTITY_DATA.encode()).digest()
TEST_VERIFICATION_HASH = hashlib.sha256(b"verification_data").digest()


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_env(monkeypatch):
    """Mock environment variables"""
    monkeypatch.setenv("SEPOLIA_RPC_URL", "http://localhost:8545")
    monkeypatch.setenv("BACKEND_PRIVATE_KEY", "0x" + "1" * 64)
    monkeypatch.setenv("CONTRACT_ADDRESS", "0x" + "2" * 40)


@pytest.fixture
def mock_web3(mocker):
    """Mock Web3 connection"""
    mock_w3 = MagicMock()
    mock_w3.is_connected.return_value = True
    mock_w3.eth.block_number = 12345
    mock_w3.eth.chain_id = 11155111
    mock_w3.eth.gas_price = 25000000000
    mock_w3.eth.get_transaction_count.return_value = 0
    
    return mock_w3


# ============================================================================
# Hash Calculation Tests
# ============================================================================

class TestHashCalculation:
    """Tests for SHA256 hash calculation"""
    
    def test_calculate_identity_hash_valid(self):
        """Test valid hash calculation"""
        data = "test_data"
        result = BlockchainService.calculate_identity_hash(data)
        
        assert isinstance(result, bytes)
        assert len(result) == 32
        
        # Verify it's correct SHA256
        expected = hashlib.sha256(data.encode()).digest()
        assert result == expected
    
    def test_calculate_identity_hash_json(self):
        """Test hash calculation with JSON data"""
        data = json.dumps({"key": "value"}, sort_keys=True)
        result = BlockchainService.calculate_identity_hash(data)
        
        assert len(result) == 32
        # Should be consistent
        result2 = BlockchainService. calculate_identity_hash(data)
        assert result == result2
    
    def test_calculate_identity_hash_empty_string(self):
        """Test hash of empty string"""
        result = BlockchainService.calculate_identity_hash("")
        assert len(result) == 32
        
        # Verify it's SHA256 of empty string
        expected = hashlib.sha256(b"").digest()
        assert result == expected
    
    def test_calculate_identity_hash_unicode(self):
        """Test hash with unicode characters"""
        data = "测试数据 🔐"
        result = BlockchainService.calculate_identity_hash(data)
        
        assert isinstance(result, bytes)
        assert len(result) == 32
    
    def test_calculate_identity_hash_invalid_type(self):
        """Test hash with non-string input"""
        with pytest.raises(TypeError):
            BlockchainService.calculate_identity_hash(123)


# ============================================================================
# Data Class Tests
# ============================================================================

class TestVerificationRecord:
    """Tests for VerificationRecord dataclass"""
    
    def test_verification_record_creation(self):
        """Test creating VerificationRecord"""
        record = VerificationRecord(
            verification_hash="0xabc123",
            verified=True,
            timestamp=1704067200
        )
        
        assert record.verification_hash == "0xabc123"
        assert record.verified is True
        assert record.timestamp == 1704067200
    
    def test_verification_record_to_dict(self):
        """Test converting VerificationRecord to dict"""
        record = VerificationRecord(
            verification_hash="0xabc123",
            verified=True,
            timestamp=1704067200
        )
        
        result = record.to_dict()
        
        assert isinstance(result, dict)
        assert result["verification_hash"] == "0xabc123"
        assert result["verified"] is True
        assert result["timestamp"] == 1704067200
        assert "timestamp_readable" in result


class TestIdentityRecord:
    """Tests for IdentityRecord dataclass"""
    
    def test_identity_record_creation(self):
        """Test creating IdentityRecord"""
        record = IdentityRecord(
            identity_hash="0x" + "1" * 64,
            created_at=1704067200,
            exists=True
        )
        
        assert record.identity_hash == "0x" + "1" * 64
        assert record. created_at == 1704067200
        assert record.exists is True
    
    def test_identity_record_to_dict(self):
        """Test converting IdentityRecord to dict"""
        record = IdentityRecord(
            identity_hash="0x" + "1" * 64,
            created_at=1704067200,
            exists=True
        )
        
        result = record.to_dict()
        
        assert isinstance(result, dict)
        assert result["exists"] is True


# ============================================================================
# Environment Validation Tests
# ============================================================================

class TestEnvironmentValidation:
    """Tests for environment variable validation"""
    
    def test_validate_env_missing_all(self, monkeypatch):
        """Test validation fails when all env vars missing"""
        monkeypatch. delenv("SEPOLIA_RPC_URL", raising=False)
        monkeypatch.delenv("BACKEND_PRIVATE_KEY", raising=False)
        monkeypatch.delenv("CONTRACT_ADDRESS", raising=False)
        
        with pytest.raises(ValueError) as excinfo:
            BlockchainService._validate_env_variables()
        
        # Check if the message contains the key phrase (case-insensitive)
        error_msg = str(excinfo. value).lower()
        assert "missing" in error_msg and "environment" in error_msg
    
    def test_validate_env_missing_one(self, monkeypatch):
        """Test validation fails when one env var missing"""
        monkeypatch.setenv("SEPOLIA_RPC_URL", "http://localhost")
        monkeypatch.setenv("BACKEND_PRIVATE_KEY", "0x123")
        monkeypatch. delenv("CONTRACT_ADDRESS", raising=False)
        
        with pytest.raises(ValueError):
            BlockchainService._validate_env_variables()
    
    def test_validate_env_all_present(self, mock_env):
        """Test validation passes when all env vars present"""
        # Should not raise
        BlockchainService._validate_env_variables()


# ============================================================================
# Input Validation Tests
# ============================================================================

class TestInputValidation: 
    """Tests for input validation methods"""
    
    def test_register_did_empty_string_hash(self):
        """Test that empty string still produces a valid hash"""
        # Empty string is technically valid for hashing (SHA256 of empty string)
        result = BlockchainService.calculate_identity_hash("")
        assert len(result) == 32
    
    def test_identify_hash_type_validation(self):
        """Test that non-string identity data is handled"""
        data = {"key": "value"}  # dict instead of string
        
        # Should raise TypeError
        with pytest.raises(TypeError):
            BlockchainService. calculate_identity_hash(data)


# ============================================================================
# Mock Integration Tests
# ============================================================================

class TestBlockchainServiceMocked:
    """Tests for BlockchainService with mocked Web3"""
    
    @pytest.fixture
    def blockchain_service(self, mocker, mock_env):
        """Create BlockchainService with mocked Web3"""
        from unittest.mock import MagicMock, patch
        
        # Mock Web3 and contract
        mock_w3 = MagicMock()
        mock_w3.is_connected.return_value = True
        mock_w3.eth.block_number = 12345
        mock_w3.eth.chain_id = 11155111
        mock_w3.eth.gas_price = 25000000000
        mock_w3.eth.get_transaction_count.return_value = 0
        mock_w3.eth.account. from_key.return_value = MagicMock(
            address="0x" + "a" * 40
        )
        
        # Mock contract
        mock_contract = MagicMock()
        
        # Patch Web3 in blockchain_service module
        with patch('blockchain_service.Web3', return_value=mock_w3):
            with patch. object(BlockchainService, '_load_contract_abi', return_value=[]):
                with patch.object(mock_w3.eth, 'contract', return_value=mock_contract):
                    service = BlockchainService()
                    service.w3 = mock_w3
                    service.contract = mock_contract
                    return service
    
    def test_initialization(self, blockchain_service):
        """Test BlockchainService initialization"""
        assert blockchain_service.rpc_url == "http://localhost:8545"
        assert blockchain_service.contract_address is not None
        assert blockchain_service. w3.is_connected() is True
    
    def test_register_did_success(self, blockchain_service):
        """Test successful DID registration"""
        # Mock transaction
        mock_tx = {
            "hash": b"tx_hash_123",
            "status": 1,
            "blockNumber":  12346,
            "gasUsed": 145230
        }
        
        blockchain_service.w3.eth.send_raw_transaction.return_value = b"tx_hash_123"
        blockchain_service.w3.eth. wait_for_transaction_receipt.return_value = mock_tx
        blockchain_service.w3.eth. account.sign_transaction.return_value = MagicMock(
            rawTransaction=b"signed_tx"
        )
        blockchain_service.contract.functions.registerDID.return_value. build_transaction.return_value = {}
        
        # Execute
        result = blockchain_service.register_did(
            did=TEST_DID,
            identity_data=TEST_IDENTITY_DATA
        )
        
        # Assert
        assert result["status"] == "success"
        assert "tx_hash" in result
        assert result["block_number"] == 12346
        assert result["gas_used"] == 145230
        assert result["did"] == TEST_DID
    
    def test_register_did_invalid_did(self, blockchain_service):
        """Test register_did with empty DID"""
        with pytest.raises(ValueError) as excinfo:
            blockchain_service. register_did(did="", identity_data=TEST_IDENTITY_DATA)
        
        assert "DID must be a non-empty string" in str(excinfo.value)
    
    def test_register_did_invalid_identity_data(self, blockchain_service):
        """Test register_did with empty identity data"""
        with pytest. raises(ValueError) as excinfo:
            blockchain_service.register_did(did=TEST_DID, identity_data="")
        
        assert "Identity data must be a non-empty string" in str(excinfo.value)
    
    def test_register_did_transaction_failed(self, blockchain_service):
        """Test register_did when transaction fails"""
        # Mock failed transaction
        mock_tx = {
            "hash": b"tx_hash_123",
            "status": 0,  # Failed
            "blockNumber":  12346,
            "gasUsed": 145230
        }
        
        blockchain_service.w3.eth.send_raw_transaction.return_value = b"tx_hash_123"
        blockchain_service.w3.eth.wait_for_transaction_receipt.return_value = mock_tx
        blockchain_service. w3.eth.account.sign_transaction.return_value = MagicMock(
            rawTransaction=b"signed_tx"
        )
        blockchain_service.contract.functions.registerDID.return_value.build_transaction.return_value = {}
        
        # Execute and assert
        with pytest.raises(Exception):
            blockchain_service.register_did(
                did=TEST_DID,
                identity_data=TEST_IDENTITY_DATA
            )
    
    def test_log_verification_success(self, blockchain_service):
        """Test successful verification logging"""
        # Mock transaction
        mock_tx = {
            "hash": b"tx_hash_456",
            "status": 1,
            "blockNumber": 12347,
            "gasUsed": 115430
        }
        
        blockchain_service.w3.eth. send_raw_transaction.return_value = b"tx_hash_456"
        blockchain_service. w3.eth.wait_for_transaction_receipt.return_value = mock_tx
        blockchain_service.w3.eth.account.sign_transaction.return_value = MagicMock(
            rawTransaction=b"signed_tx"
        )
        blockchain_service.contract.functions.logVerification.return_value.build_transaction.return_value = {}
        
        # Execute
        result = blockchain_service.log_verification(
            verification_hash=TEST_VERIFICATION_HASH,
            did=TEST_DID,
            verified=True,
            final_score=0.91,
            verification_threshold=0.7
        )
        
        # Assert
        assert result["status"] == "success"
        assert "tx_hash" in result
        assert result["verified"] is True
        assert result["final_score"] == 0.91
        assert result["block_number"] == 12347
    
    def test_log_verification_below_threshold(self, blockchain_service):
        """Test verification logging with score below threshold"""
        # Mock transaction
        mock_tx = {
            "hash": b"tx_hash_456",
            "status": 1,
            "blockNumber": 12347,
            "gasUsed": 115430
        }
        
        blockchain_service. w3.eth.send_raw_transaction.return_value = b"tx_hash_456"
        blockchain_service.w3.eth.wait_for_transaction_receipt.return_value = mock_tx
        blockchain_service.w3.eth.account.sign_transaction.return_value = MagicMock(
            rawTransaction=b"signed_tx"
        )
        blockchain_service. contract.functions.logVerification. return_value.build_transaction. return_value = {}
        
        # Execute with score below threshold
        result = blockchain_service.log_verification(
            verification_hash=TEST_VERIFICATION_HASH,
            did=TEST_DID,
            verified=False,
            final_score=0.65,
            verification_threshold=0.7
        )
        
        # Assert
        assert result["verified"] is False
        assert result["final_score"] == 0.65
    
    def test_log_verification_invalid_hash(self, blockchain_service):
        """Test log_verification with invalid hash"""
        # Hash too short
        invalid_hash = b"short_hash"
        
        with pytest.raises(ValueError) as excinfo:
            blockchain_service.log_verification(
                verification_hash=invalid_hash,
                did=TEST_DID,
                verified=True
            )
        
        assert "32 bytes" in str(excinfo.value)
    
    def test_log_verification_invalid_did(self, blockchain_service):
        """Test log_verification with invalid DID"""
        with pytest.raises(ValueError) as excinfo:
            blockchain_service. log_verification(
                verification_hash=TEST_VERIFICATION_HASH,
                did="",
                verified=True
            )
        
        assert "DID must be a non-empty string" in str(excinfo.value)
    
    def test_get_on_chain_history_success(self, blockchain_service):
        """Test successful history retrieval"""
        # Mock contract response
        mock_verifications = [
            (TEST_VERIFICATION_HASH, True, 1704067200),
            (TEST_VERIFICATION_HASH, True, 1704067300),
        ]
        
        blockchain_service.contract.functions.getVerifications.return_value.call.return_value = mock_verifications
        
        # Execute
        result = blockchain_service.get_on_chain_history(TEST_DID)
        
        # Assert
        assert isinstance(result, list)
        assert len(result) == 2
        assert all(isinstance(r, VerificationRecord) for r in result)
        assert result[0]. verified is True
        assert result[0].timestamp == 1704067200
    
    def test_get_on_chain_history_empty(self, blockchain_service):
        """Test history retrieval for DID with no verifications"""
        # Mock empty response
        blockchain_service.contract. functions.getVerifications.return_value.call.return_value = []
        
        # Execute
        result = blockchain_service.get_on_chain_history(TEST_DID)
        
        # Assert
        assert isinstance(result, list)
        assert len(result) == 0
    
    def test_get_on_chain_history_invalid_did(self, blockchain_service):
        """Test history retrieval with invalid DID"""
        with pytest.raises(ValueError) as excinfo:
            blockchain_service.get_on_chain_history("")
        
        assert "DID must be a non-empty string" in str(excinfo.value)
    
    def test_get_identity_record_success(self, blockchain_service):
        """Test successful identity record retrieval"""
        # Mock contract response
        mock_identity = (TEST_IDENTITY_HASH, 1704067200, True)
        
        blockchain_service.contract.functions.getIdentity.return_value.call.return_value = mock_identity
        
        # Execute
        result = blockchain_service.get_identity_record(TEST_DID)
        
        # Assert
        assert isinstance(result, IdentityRecord)
        assert result.created_at == 1704067200
        assert result.exists is True
    
    def test_get_identity_record_invalid_did(self, blockchain_service):
        """Test identity record retrieval with invalid DID"""
        with pytest.raises(ValueError):
            blockchain_service.get_identity_record("")
    
    def test_did_exists_true(self, blockchain_service):
        """Test did_exists returns True"""
        blockchain_service.contract.functions.didExists.return_value.call. return_value = True
        
        result = blockchain_service.did_exists(TEST_DID)
        
        assert result is True
    
    def test_did_exists_false(self, blockchain_service):
        """Test did_exists returns False"""
        blockchain_service. contract.functions.didExists.return_value.call.return_value = False
        
        result = blockchain_service.did_exists(TEST_DID)
        
        assert result is False
    
    def test_did_exists_invalid_did(self, blockchain_service):
        """Test did_exists with invalid DID"""
        with pytest.raises(ValueError):
            blockchain_service.did_exists("")
    
    def test_get_verification_count_success(self, blockchain_service):
        """Test successful verification count retrieval"""
        blockchain_service.contract.functions.getVerificationCount.return_value. call.return_value = 5
        
        result = blockchain_service.get_verification_count(TEST_DID)
        
        assert result == 5
        assert isinstance(result, int)
    
    def test_get_verification_count_zero(self, blockchain_service):
        """Test verification count when none exist"""
        blockchain_service. contract.functions.getVerificationCount.return_value.call.return_value = 0
        
        result = blockchain_service.get_verification_count(TEST_DID)
        
        assert result == 0
    
    def test_get_verification_count_invalid_did(self, blockchain_service):
        """Test verification count with invalid DID"""
        with pytest.raises(ValueError):
            blockchain_service.get_verification_count("")
    
    def test_estimate_registration_gas(self, blockchain_service):
        """Test gas estimation for registration"""
        blockchain_service.contract.functions.registerDID.return_value.estimate_gas.return_value = 145230
        
        result = blockchain_service.estimate_registration_gas(
            did=TEST_DID,
            identity_data=TEST_IDENTITY_DATA
        )
        
        assert result == 145230
        assert isinstance(result, int)
    
    def test_estimate_verification_gas(self, blockchain_service):
        """Test gas estimation for verification"""
        blockchain_service. contract.functions.logVerification. return_value.estimate_gas. return_value = 115430
        
        result = blockchain_service.estimate_verification_gas(
            verification_hash=TEST_VERIFICATION_HASH,
            did=TEST_DID,
            verified=True
        )
        
        assert result == 115430
        assert isinstance(result, int)


# ============================================================================
# Edge Cases Tests
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions"""
    
    def test_very_long_did(self):
        """Test with very long DID string"""
        long_did = "did:example:" + "x" * 1000
        result = BlockchainService.calculate_identity_hash(long_did)
        
        assert len(result) == 32
    
    def test_special_characters_in_did(self):
        """Test DID with special characters"""
        special_did = "did:example: test! @#$%^&*()"
        result = BlockchainService.calculate_identity_hash(special_did)
        
        assert len(result) == 32
    
    def test_large_json_identity_data(self):
        """Test with large JSON identity data"""
        large_data = json.dumps({
            "field_" + str(i): "value_" * 100
            for i in range(100)
        }, sort_keys=True)
        
        result = BlockchainService. calculate_identity_hash(large_data)
        
        assert len(result) == 32


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__": 
    pytest.main([__file__, "-v", "--tb=short"])