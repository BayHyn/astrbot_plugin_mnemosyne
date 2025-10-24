# -*- coding: utf-8 -*-
"""
记忆管理相关数据模型
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from datetime import datetime


@dataclass
class MemoryRecord:
    """记忆记录"""
    memory_id: str
    session_id: str
    content: str
    create_time: datetime
    persona_id: Optional[str] = None
    similarity_score: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "memory_id": self.memory_id,
            "session_id": self.session_id,
            "content": self.content,
            "create_time": self.create_time.isoformat() if isinstance(self.create_time, datetime) else str(self.create_time),
            "persona_id": self.persona_id,
            "similarity_score": self.similarity_score,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'MemoryRecord':
        """从字典创建实例"""
        create_time = data.get('create_time')
        if isinstance(create_time, str):
            try:
                create_time = datetime.fromisoformat(create_time)
            except ValueError:
                create_time = datetime.now()
        elif isinstance(create_time, (int, float)):
            create_time = datetime.fromtimestamp(create_time)
        
        return cls(
            memory_id=data.get('memory_id', ''),
            session_id=data.get('session_id', ''),
            content=data.get('content', ''),
            create_time=create_time or datetime.now(),
            persona_id=data.get('persona_id'),
            similarity_score=data.get('similarity_score'),
            metadata=data.get('metadata', {})
        )


@dataclass
class MemoryStatistics:
    """记忆统计信息"""
    total_memories: int = 0
    total_sessions: int = 0
    memories_by_session: Dict[str, int] = field(default_factory=dict)
    memories_by_date: Dict[str, int] = field(default_factory=dict)
    most_active_sessions: List[tuple] = field(default_factory=list)
    recent_memories_count: int = 0
    average_memory_length: float = 0.0
    
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "total_memories": self.total_memories,
            "total_sessions": self.total_sessions,
            "memories_by_session": self.memories_by_session,
            "memories_by_date": self.memories_by_date,
            "most_active_sessions": [
                {"session_id": session_id, "count": count}
                for session_id, count in self.most_active_sessions
            ],
            "recent_memories_count": self.recent_memories_count,
            "average_memory_length": round(self.average_memory_length, 2),
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class MemorySearchRequest:
    """记忆搜索请求"""
    session_id: Optional[str] = None
    keyword: Optional[str] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    persona_id: Optional[str] = None
    limit: int = 10
    offset: int = 0
    sort_by: str = "create_time"  # create_time, similarity
    sort_order: str = "desc"  # asc, desc


@dataclass
class MemorySearchResponse:
    """记忆搜索响应"""
    records: List[MemoryRecord]
    total_count: int
    page: int
    page_size: int
    has_more: bool
    
    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "records": [record.to_dict() for record in self.records],
            "total_count": self.total_count,
            "page": self.page,
            "page_size": self.page_size,
            "has_more": self.has_more
        }