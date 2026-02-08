"""
Chat management backend for Document Q&A

This module handles chat state management, message storage, and document processing
without any UI dependencies.
"""

import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from loguru import logger

from .config import DEFAULT_RAG_CONFIG


@dataclass
class ChatMessage:
    """Represents a single chat message"""
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass 
class ChatSession:
    """Represents a complete chat session"""
    id: str
    title: str
    messages: List[ChatMessage] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    document_name: Optional[str] = None
    document_content: Optional[bytes] = None
    document_text: str = ""
    document_id: Optional[str] = None  # RAG document ID
    rag_processed: bool = False
    rag_stats: Optional[Dict[str, Any]] = None
    rag_backend: str = "vector"  # "vector" or "pageindex"
    
    def add_message(self, role: str, content: str) -> None:
        """Add a message to this chat session"""
        message = ChatMessage(role=role, content=content)
        self.messages.append(message)
        
        # Update title from first user message
        if role == "user" and self.title == "New Chat":
            self.title = content[:50] + ("..." if len(content) > 50 else "")
    
    def get_message_history(self) -> List[Dict[str, str]]:
        """Get message history in dict format for LLM consumption"""
        return [{"role": msg.role, "content": msg.content} for msg in self.messages]


class ChatManager:
    """Backend chat management without UI dependencies"""
    
    def __init__(self):
        """Initialize chat manager"""
        self.chats: Dict[str, ChatSession] = {}
        self.current_chat_id: Optional[str] = None
        self._default_rag_config = DEFAULT_RAG_CONFIG.copy()
    
    def create_new_chat(self, clear_rag_callback: Optional[callable] = None) -> str:
        """
        Create a new chat session
        
        Args:
            clear_rag_callback: Optional callback to clear RAG system
            
        Returns:
            New chat ID
        """
        chat_id = str(uuid.uuid4())
        chat_session = ChatSession(
            id=chat_id,
            title="New Chat"
        )
        
        self.chats[chat_id] = chat_session
        self.current_chat_id = chat_id
        
        # Clear RAG system if callback provided
        if clear_rag_callback:
            try:
                clear_rag_callback()
                logger.info(f"Cleared RAG system for new chat: {chat_id}")
            except Exception as e:
                logger.warning(f"Could not clear RAG system: {e}")
        
        return chat_id
    
    def get_current_chat(self) -> Optional[ChatSession]:
        """Get current chat session"""
        if self.current_chat_id:
            return self.chats.get(self.current_chat_id)
        return None
    
    def get_chat(self, chat_id: str) -> Optional[ChatSession]:
        """Get specific chat session by ID"""
        return self.chats.get(chat_id)
    
    def switch_to_chat(self, chat_id: str) -> bool:
        """Switch to a different chat"""
        if chat_id in self.chats:
            self.current_chat_id = chat_id
            return True
        return False
    
    def delete_chat(self, chat_id: str) -> bool:
        """
        Delete a chat session
        
        Args:
            chat_id: ID of chat to delete
            
        Returns:
            True if deleted successfully
        """
        if chat_id not in self.chats:
            return False
            
        del self.chats[chat_id]
        
        # If we deleted the current chat, switch to another or create new
        if self.current_chat_id == chat_id:
            if self.chats:
                # Switch to the most recent chat
                sorted_chats = sorted(
                    self.chats.items(),
                    key=lambda x: x[1].created_at,
                    reverse=True
                )
                self.current_chat_id = sorted_chats[0][0]
            else:
                # No chats left, create a new one
                self.create_new_chat()
        
        return True
    
    def add_message_to_current(self, role: str, content: str) -> bool:
        """Add message to current chat"""
        current_chat = self.get_current_chat()
        if current_chat:
            current_chat.add_message(role, content)
            return True
        return False
    
    def update_document(
        self, 
        document_name: str, 
        document_content: bytes, 
        document_text: str,
        chat_id: Optional[str] = None
    ) -> bool:
        """
        Update document for a chat session
        
        Args:
            document_name: Name of the document
            document_content: Binary content of the document
            document_text: Extracted text content
            chat_id: Chat ID (uses current if None)
            
        Returns:
            True if updated successfully
        """
        target_chat_id = chat_id or self.current_chat_id
        if not target_chat_id or target_chat_id not in self.chats:
            return False
            
        chat = self.chats[target_chat_id]
        chat.document_name = document_name
        chat.document_content = document_content
        chat.document_text = document_text
        chat.title = f"Doc: {document_name}"
        chat.rag_processed = False
        chat.rag_stats = None
        
        return True
    
    def update_rag_processing(
        self, 
        rag_stats: Dict[str, Any], 
        chat_id: Optional[str] = None
    ) -> bool:
        """
        Update RAG processing status for a chat
        
        Args:
            rag_stats: Statistics from RAG processing
            chat_id: Chat ID (uses current if None)
            
        Returns:
            True if updated successfully
        """
        target_chat_id = chat_id or self.current_chat_id
        if not target_chat_id or target_chat_id not in self.chats:
            return False
            
        chat = self.chats[target_chat_id]
        chat.rag_processed = True
        chat.rag_stats = rag_stats
        chat.document_id = rag_stats.get("document_id")  # Store the RAG document ID
        
        return True
    
    def get_sorted_chats(self) -> List[tuple[str, ChatSession]]:
        """Get chats sorted by creation date (newest first)"""
        return sorted(
            self.chats.items(),
            key=lambda x: x[1].created_at,
            reverse=True
        )
    
    def has_valid_document(self, chat_id: Optional[str] = None) -> bool:
        """Check if chat has valid document content"""
        target_chat_id = chat_id or self.current_chat_id
        if not target_chat_id or target_chat_id not in self.chats:
            return False
            
        chat = self.chats[target_chat_id]
        return bool(chat.document_text and chat.document_text.strip())
    
    def get_chat_count(self) -> int:
        """Get total number of chats"""
        return len(self.chats)
    
    def clear_all_chats(self) -> None:
        """Clear all chats and create a new one"""
        self.chats.clear()
        self.current_chat_id = None
        self.create_new_chat() 