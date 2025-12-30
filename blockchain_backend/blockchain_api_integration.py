"""
FastAPI Integration Example for Blockchain Service
Demonstrates how to integrate the Blockchain Service module with FastAPI endpoints. 

Author: Backend Engineer
Date: 2025-12-30
"""

import json
from hashlib import sha256
from typing import Dict, List, Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from blockchain_service import BlockchainService, VerificationRecord


# ============================================================================
# Pydantic Models for API Request/Response
# ============================================================================

class IdentityVerificationRequest(BaseModel):
    """Request model for DID registration"""
    did: str = Field(..., description="Decentralized Identifier")
    face_score: float = Field(..., ge=0, le=1, description="Face recognition score")
    voice_score: float = Field(..., ge=0, le=1, description="Voice biometric score")
    document_score: float = Field(..., ge=0, le=1, description="Document verification score")
    face_weight: float = Field(default=0.4, ge=0, le=1)
    voice_weight: float = Field(default=0.35, ge=0, le=1)
    document_weight: float = Field(default=0.25, ge=0, le=1)


class VerificationLogRequest(BaseModel):
    """Request model for logging verification"""
    did: str = Field(..., description="Decentralized Identifier")
    verification_hash: str = Field(..., description="32-byte verification hash (hex)")
    final_score: float = Field(..., ge=0, le=1, description="Final verification score")
    threshold: float = Field(default=0.7, ge=0, le=1, description="Verification threshold")


class DIDRegistrationResponse(BaseModel):
    """Response model for DID registration"""
    success: bool
    message: str
    did: str
    tx_hash: str
    identity_hash: str
    block_number: int
    gas_used: int


class VerificationLogResponse(BaseModel):
    """Response model for verification logging"""
    success: bool
    message: str
    did: str
    tx_hash: str
    verified: bool
    final_score: float
    block_number: int
    gas_used: int


class VerificationHistory(BaseModel):
    """Response model for verification history"""
    did: str
    total_verifications: int
    records: List[Dict]


# ============================================================================
# FastAPI Application Setup
# ============================================================================

app = FastAPI(
    title="DID++ Blockchain Service API",
    description="FastAPI integration for DID++ blockchain operations",
    version="1.0.0"
)

# Initialize Blockchain Service
blockchain_service = BlockchainService()


# ============================================================================
# Helper Functions
# ============================================================================

def calculate_weighted_score(
    face_score: float,
    voice_score: float,
    document_score: float,
    face_weight: float = 0.4,
    voice_weight: float = 0.35,
    document_weight: float = 0.25
) -> float:
    """
    Calculate weighted multimodal verification score.
    
    Args:
        face_score: Face recognition score (0-1)
        voice_score: Voice biometric score (0-1)
        document_score: Document verification score (0-1)
        face_weight: Weight for face score
        voice_weight: Weight for voice score
        document_weight: Weight for document score
    
    Returns:
        float: Weighted final score (0-1)
    """
    total_weight = face_weight + voice_weight + document_weight
    
    if total_weight != 1.0:
        raise ValueError("Weights must sum to 1.0")
    
    final_score = (
        face_score * face_weight +
        voice_score * voice_weight +
        document_score * document_weight
    )
    
    return final_score


def create_identity_hash_input(
    face_score: float,
    voice_score: float,
    document_score: float,
    did: str
) -> str:
    """
    Create identity hash input from multimodal scores (Stage 3 of pipeline).
    
    Args:
        face_score: Face recognition score
        voice_score: Voice biometric score
        document_score: Document verification score
        did:  Decentralized Identifier
    
    Returns:
        str:  JSON string of scores to be hashed
    """
    identity_data = {
        "did": did,
        "face_score": face_score,
        "voice_score": voice_score,
        "document_score":  document_score,
        "timestamp": int(__import__("time").time())
    }
    
    return json.dumps(identity_data, sort_keys=True)


def create_verification_hash_input(
    did: str,
    face_score: float,
    voice_score: float,
    document_score: float,
    final_score: float,
    verified: bool
) -> str:
    """
    Create verification hash input from verification results. 
    
    Args:
        did: Decentralized Identifier
        face_score: Face recognition score
        voice_score: Voice biometric score
        document_score: Document verification score
        final_score: Final weighted score
        verified: Verification result
    
    Returns:
        str: JSON string to be hashed
    """
    verification_data = {
        "did": did,
        "face_score": face_score,
        "voice_score": voice_score,
        "document_score": document_score,
        "final_score": final_score,
        "verified": verified,
        "timestamp": int(__import__("time").time())
    }
    
    return json.dumps(verification_data, sort_keys=True)


# ============================================================================
# Health Check Endpoints
# ============================================================================

@app.get("/health", tags=["Health"])
async def health_check() -> Dict:
    """Check service health and blockchain connectivity"""
    try:
        block_number = blockchain_service.w3.eth.block_number
        chain_id = blockchain_service.w3.eth.chain_id
        
        return {
            "status": "healthy",
            "blockchain_connected": True,
            "current_block": block_number,
            "chain_id": chain_id,
            "chain_name": "Sepolia" if chain_id == 11155111 else "Unknown",
            "account_address": blockchain_service.account_address
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service unavailable: {str(e)}")


# ============================================================================
# DID Registration Endpoints (Flow 1)
# ============================================================================

@app.post("/api/v1/register-did", response_model=DIDRegistrationResponse, tags=["Registration"])
async def register_did(request: IdentityVerificationRequest) -> DIDRegistrationResponse: 
    """
    Register a new DID on the blockchain.
    
    Flow 1: Calls contract. registerDID with calculated identity hash. 
    - Validates input scores and weights
    - Creates identity hash from multimodal scores
    - Signs transaction with backend private key
    - Waits for confirmation (~150,000 gas)
    - Returns transaction hash and block number
    
    Args:
        request: IdentityVerificationRequest containing DID and verification scores
    
    Returns: 
        DIDRegistrationResponse with transaction details
    
    Raises:
        HTTPException: If registration fails
    """
    try:
        # Validate weights sum to 1.0
        total_weight = (
            request.face_weight +
            request.voice_weight +
            request.document_weight
        )
        
        if abs(total_weight - 1.0) > 0.001:
            raise HTTPException(
                status_code=400,
                detail="Weights must sum to 1.0"
            )
        
        # Check if DID already exists
        if blockchain_service.did_exists(request.did):
            raise HTTPException(
                status_code=409,
                detail=f"DID {request.did} is already registered"
            )
        
        # Create identity hash input from multimodal scores (Stage 3)
        identity_data = create_identity_hash_input(
            face_score=request.face_score,
            voice_score=request.voice_score,
            document_score=request.document_score,
            did=request.did
        )
        
        # Register DID on blockchain
        tx_result = blockchain_service.register_did(
            did=request.did,
            identity_data=identity_data,
            gas_limit=150000
        )
        
        return DIDRegistrationResponse(
            success=True,
            message=f"DID {request.did} registered successfully",
            did=request.did,
            tx_hash=tx_result["tx_hash"],
            identity_hash=tx_result["identity_hash"],
            block_number=tx_result["block_number"],
            gas_used=tx_result["gas_used"]
        )
    
    except Exception as e: 
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/v1/did/{did}/exists", tags=["Registration"])
async def check_did_exists(did: str) -> Dict:
    """Check if a DID is already registered"""
    try:
        exists = blockchain_service.did_exists(did)
        
        if exists:
            identity = blockchain_service.get_identity_record(did)
            return {
                "did": did,
                "exists": True,
                "identity": identity. to_dict()
            }
        else:
            return {
                "did": did,
                "exists": False,
                "identity": None
            }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ============================================================================
# Verification Logging Endpoints (Flow 2)
# ============================================================================

@app. post("/api/v1/log-verification", response_model=VerificationLogResponse, tags=["Verification"])
async def log_verification(request:  VerificationLogRequest) -> VerificationLogResponse:
    """
    Log verification result on the blockchain.
    
    Flow 2: Calls contract.logVerification with verification hash.
    - Validates verification hash and scores
    - Only logs if final_score meets threshold (verified = True)
    - Signs transaction with backend private key
    - Waits for confirmation (~120,000 gas)
    - Returns transaction hash and block number
    
    Args:
        request:  VerificationLogRequest with DID and verification results
    
    Returns:
        VerificationLogResponse with transaction details
    
    Raises: 
        HTTPException: If logging fails
    """
    try:
        # Convert hex verification hash to bytes
        verification_hash_bytes = bytes.fromhex(request.verification_hash. lstrip("0x"))
        
        if len(verification_hash_bytes) != 32:
            raise HTTPException(
                status_code=400,
                detail="Verification hash must be 32 bytes (64 hex characters)"
            )
        
        # Determine verification status based on threshold
        verified = request.final_score >= request.threshold
        
        # Log verification on blockchain
        tx_result = blockchain_service.log_verification(
            verification_hash=verification_hash_bytes,
            did=request.did,
            verified=verified,
            final_score=request.final_score,
            verification_threshold=request.threshold,
            gas_limit=120000
        )
        
        return VerificationLogResponse(
            success=True,
            message=f"Verification for {request.did} logged successfully",
            did=request.did,
            tx_hash=tx_result["tx_hash"],
            verified=verified,
            final_score=request.final_score,
            block_number=tx_result["block_number"],
            gas_used=tx_result["gas_used"]
        )
    
    except Exception as e: 
        raise HTTPException(status_code=400, detail=str(e))


# ============================================================================
# History Query Endpoints (Flow 3)
# ============================================================================

@app. get("/api/v1/did/{did}/history", response_model=VerificationHistory, tags=["History"])
async def get_did_history(did:  str) -> VerificationHistory: 
    """
    Retrieve on-chain verification history for a DID.
    
    Flow 3: Calls contract.getVerifications(did).
    - Returns list of all verification events
    - Includes verification hash, verified status, and block timestamp
    - No gas cost (read-only query)
    
    Args:
        did:  Decentralized Identifier
    
    Returns:
        VerificationHistory containing all verification records
    
    Raises: 
        HTTPException: If query fails
    """
    try:
        # Check if DID exists
        if not blockchain_service.did_exists(did):
            raise HTTPException(
                status_code=404,
                detail=f"DID {did} not found on blockchain"
            )
        
        # Get verification count
        count = blockchain_service.get_verification_count(did)
        
        # Get verification history
        records = blockchain_service.get_on_chain_history(did)
        
        # Convert records to dictionaries
        records_dict = [record.to_dict() for record in records]
        
        return VerificationHistory(
            did=did,
            total_verifications=count,
            records=records_dict
        )
    
    except HTTPException: 
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/v1/did/{did}/verification-count", tags=["History"])
async def get_verification_count(did: str) -> Dict:
    """Get total verification count for a DID"""
    try:
        if not blockchain_service.did_exists(did):
            raise HTTPException(
                status_code=404,
                detail=f"DID {did} not found on blockchain"
            )
        
        count = blockchain_service.get_verification_count(did)
        
        return {
            "did":  did,
            "total_verifications": count
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ============================================================================
# Gas Estimation Endpoints
# ============================================================================

@app. post("/api/v1/estimate/registration-gas", tags=["Gas Estimation"])
async def estimate_registration_gas(request: IdentityVerificationRequest) -> Dict:
    """Estimate gas cost for DID registration"""
    try:
        identity_data = create_identity_hash_input(
            face_score=request.face_score,
            voice_score=request.voice_score,
            document_score=request.document_score,
            did=request.did
        )
        
        estimated_gas = blockchain_service.estimate_registration_gas(
            did=request.did,
            identity_data=identity_data
        )
        
        gas_price = blockchain_service.base_gas_price
        estimated_cost_wei = estimated_gas * gas_price
        estimated_cost_eth = blockchain_service.w3.from_wei(estimated_cost_wei, "ether")
        
        return {
            "operation": "register_did",
            "estimated_gas": estimated_gas,
            "gas_price_wei": gas_price,
            "estimated_cost_wei": estimated_cost_wei,
            "estimated_cost_eth": str(estimated_cost_eth)
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/v1/estimate/verification-gas", tags=["Gas Estimation"])
async def estimate_verification_gas(request: VerificationLogRequest) -> Dict:
    """Estimate gas cost for verification logging"""
    try:
        verification_hash_bytes = bytes.fromhex(request.verification_hash.lstrip("0x"))
        verified = request.final_score >= request.threshold
        
        estimated_gas = blockchain_service.estimate_verification_gas(
            verification_hash=verification_hash_bytes,
            did=request.did,
            verified=verified
        )
        
        gas_price = blockchain_service.base_gas_price
        estimated_cost_wei = estimated_gas * gas_price
        estimated_cost_eth = blockchain_service.w3.from_wei(estimated_cost_wei, "ether")
        
        return {
            "operation":  "log_verification",
            "estimated_gas": estimated_gas,
            "gas_price_wei": gas_price,
            "estimated_cost_wei": estimated_cost_wei,
            "estimated_cost_eth": str(estimated_cost_eth)
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ============================================================================
# Root Endpoint
# ============================================================================

@app.get("/", tags=["Info"])
async def root() -> Dict:
    """API root endpoint with documentation links"""
    return {
        "service": "DID++ Blockchain Service API",
        "version": "1.0.0",
        "docs": "/docs",
        "openapi_schema": "/openapi.json",
        "flows": {
            "flow_1": "DID Registration - POST /api/v1/register-did",
            "flow_2":  "Verification Logging - POST /api/v1/log-verification",
            "flow_3": "History Query - GET /api/v1/did/{did}/history"
        }
    }


if __name__ == "__main__": 
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )