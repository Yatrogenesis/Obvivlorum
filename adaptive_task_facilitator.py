#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Adaptive Task Facilitator for AI Symbiote
=========================================

This module implements an intelligent task facilitation system that learns from user
behavior, adapts to preferences, and proactively assists with task completion.

Author: Francisco Molina
ORCID: https://orcid.org/0009-0008-6093-8267
Email: pako.molina@gmail.com
"""

import os
import sys
import json
import time
import logging
import threading
import queue
import hashlib
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict, deque
import pickle

logger = logging.getLogger("AdaptiveTaskFacilitator")

@dataclass
class Task:
    """Represents a task in the system."""
    id: str
    name: str
    description: str
    priority: int = 5  # 1-10 scale
    estimated_duration: int = 30  # minutes
    dependencies: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    status: str = "pending"  # pending, in_progress, completed, cancelled
    progress: float = 0.0  # 0.0 to 1.0
    
@dataclass
class UserBehaviorPattern:
    """Represents a user behavior pattern."""
    pattern_id: str
    description: str
    frequency: int = 0
    success_rate: float = 0.0
    time_of_day: List[int] = field(default_factory=list)  # Hours when pattern occurs
    context_factors: Dict[str, Any] = field(default_factory=dict)
    last_observed: datetime = field(default_factory=datetime.now)

@dataclass
class TaskSuggestion:
    """Represents a task suggestion from the system."""
    suggestion_id: str
    task_name: str
    reasoning: str
    confidence: float  # 0.0 to 1.0
    priority: int
    estimated_benefit: float
    suggested_time: Optional[datetime] = None

class AdaptiveTaskFacilitator:
    """
    Intelligent task facilitation system that learns and adapts to user behavior.
    """
    
    def __init__(self, user_id: str = "default", data_dir: str = None):
        """
        Initialize the Adaptive Task Facilitator.
        
        Args:
            user_id: Unique identifier for the user
            data_dir: Directory to store user data and patterns
        """
        self.user_id = user_id
        self.data_dir = data_dir or os.path.join(os.path.expanduser("~"), ".ai_symbiote", "task_data")
        
        # Ensure data directory exists
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Core data structures
        self.tasks: Dict[str, Task] = {}
        self.completed_tasks: Dict[str, Task] = {}
        self.behavior_patterns: Dict[str, UserBehaviorPattern] = {}
        self.task_history: deque = deque(maxlen=1000)
        self.user_preferences: Dict[str, Any] = {}
        self.active_contexts: Dict[str, Any] = {}
        
        # Learning and adaptation
        self.pattern_detector = PatternDetector()
        self.task_predictor = TaskPredictor()
        self.suggestion_engine = SuggestionEngine()
        
        # Processing queues
        self.task_queue = queue.PriorityQueue()
        self.suggestion_queue = queue.Queue()
        
        # Threading and monitoring
        self.is_active = False
        self.monitoring_thread = None
        self.learning_thread = None
        
        # Configuration
        self.config = {
            "learning_enabled": True,
            "proactive_suggestions": True,
            "adaptive_scheduling": True,
            "context_awareness": True,
            "max_suggestions_per_hour": 5,
            "min_confidence_threshold": 0.6,
            "pattern_detection_interval": 300,  # 5 minutes
            "suggestion_cooldown": 1800,  # 30 minutes
            "auto_prioritize": True,
            "interruption_tolerance": 0.7  # How tolerant to interruptions (0-1)
        }
        
        # Load existing data
        self._load_user_data()
        
        logger.info(f"Adaptive Task Facilitator initialized for user: {user_id}")
    
    def start(self):
        """Start the adaptive task facilitation system."""
        if self.is_active:
            logger.warning("Task facilitator already active")
            return
        
        self.is_active = True
        
        # Start monitoring thread
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        # Start learning thread
        if self.config["learning_enabled"]:
            self.learning_thread = threading.Thread(target=self._learning_loop, daemon=True)
            self.learning_thread.start()
        
        logger.info("Adaptive Task Facilitator started")
    
    def stop(self):
        """Stop the task facilitation system."""
        self.is_active = False
        
        # Save user data
        self._save_user_data()
        
        logger.info("Adaptive Task Facilitator stopped")
    
    def add_task(
        self, 
        name: str, 
        description: str = "",
        priority: int = 5,
        estimated_duration: int = 30,
        dependencies: List[str] = None,
        tags: List[str] = None,
        context: Dict[str, Any] = None
    ) -> str:
        """
        Add a new task to the system.
        
        Args:
            name: Task name
            description: Task description
            priority: Task priority (1-10)
            estimated_duration: Estimated duration in minutes
            dependencies: List of task IDs this task depends on
            tags: List of tags for categorization
            context: Additional context information
            
        Returns:
            Task ID
        """
        task_id = self._generate_task_id(name)
        
        task = Task(
            id=task_id,
            name=name,
            description=description,
            priority=priority,
            estimated_duration=estimated_duration,
            dependencies=dependencies or [],
            tags=tags or [],
            context=context or {}
        )
        
        self.tasks[task_id] = task
        
        # Add to processing queue
        self.task_queue.put((-priority, time.time(), task_id))  # Negative priority for max-heap behavior
        
        # Record task creation
        self._record_user_action("task_created", {
            "task_id": task_id,
            "name": name,
            "priority": priority,
            "tags": tags or [],
            "context": context or {}
        })
        
        logger.info(f"Task added: {name} (ID: {task_id})")
        
        # Trigger adaptive suggestions if enabled
        if self.config["proactive_suggestions"]:
            self._generate_contextual_suggestions(task)
        
        return task_id
    
    def update_task_progress(self, task_id: str, progress: float, status: str = None):
        """
        Update task progress and status.
        
        Args:
            task_id: Task identifier
            progress: Progress percentage (0.0 to 1.0)
            status: Optional status update
        """
        if task_id not in self.tasks:
            logger.error(f"Task not found: {task_id}")
            return
        
        task = self.tasks[task_id]
        old_progress = task.progress
        old_status = task.status
        
        task.progress = min(1.0, max(0.0, progress))
        if status:
            task.status = status
        
        # Auto-complete task if progress reaches 100%
        if task.progress >= 1.0 and task.status != "completed":
            task.status = "completed"
            self.complete_task(task_id)
        
        # Record progress update
        self._record_user_action("task_progress_updated", {
            "task_id": task_id,
            "old_progress": old_progress,
            "new_progress": task.progress,
            "old_status": old_status,
            "new_status": task.status
        })
        
        logger.debug(f"Task progress updated: {task.name} -> {task.progress:.1%}")
    
    def complete_task(self, task_id: str, completion_notes: str = ""):
        """
        Mark a task as completed.
        
        Args:
            task_id: Task identifier
            completion_notes: Optional completion notes
        """
        if task_id not in self.tasks:
            logger.error(f"Task not found: {task_id}")
            return
        
        task = self.tasks[task_id]
        task.status = "completed"
        task.progress = 1.0
        
        # Move to completed tasks
        self.completed_tasks[task_id] = task
        del self.tasks[task_id]
        
        # Record completion
        completion_time = datetime.now()
        actual_duration = (completion_time - task.created_at).total_seconds() / 60  # minutes
        
        self._record_user_action("task_completed", {
            "task_id": task_id,
            "name": task.name,
            "estimated_duration": task.estimated_duration,
            "actual_duration": actual_duration,
            "priority": task.priority,
            "tags": task.tags,
            "completion_notes": completion_notes
        })
        
        logger.info(f"Task completed: {task.name}")
        
        # Learn from completion patterns
        if self.config["learning_enabled"]:
            self._learn_from_completion(task, actual_duration)
        
        # Check for dependent tasks
        self._check_dependent_tasks(task_id)
    
    def get_task_suggestions(self, max_suggestions: int = 5) -> List[TaskSuggestion]:
        """
        Get intelligent task suggestions based on current context and patterns.
        
        Args:
            max_suggestions: Maximum number of suggestions to return
            
        Returns:
            List of task suggestions
        """
        suggestions = []
        
        # Generate suggestions based on different strategies
        
        # 1. Context-based suggestions
        context_suggestions = self._generate_context_suggestions()
        suggestions.extend(context_suggestions)
        
        # 2. Pattern-based suggestions
        pattern_suggestions = self._generate_pattern_suggestions()
        suggestions.extend(pattern_suggestions)
        
        # 3. Dependency-based suggestions
        dependency_suggestions = self._generate_dependency_suggestions()
        suggestions.extend(dependency_suggestions)
        
        # 4. Time-based suggestions
        time_suggestions = self._generate_time_based_suggestions()
        suggestions.extend(time_suggestions)
        
        # Filter by confidence threshold
        filtered_suggestions = [
            s for s in suggestions 
            if s.confidence >= self.config["min_confidence_threshold"]
        ]
        
        # Sort by confidence and priority
        filtered_suggestions.sort(
            key=lambda x: (x.confidence, x.priority, x.estimated_benefit),
            reverse=True
        )
        
        # Return top suggestions
        return filtered_suggestions[:max_suggestions]
    
    def get_optimal_schedule(self, time_window_hours: int = 8) -> List[Dict[str, Any]]:
        """
        Generate an optimal schedule for pending tasks.
        
        Args:
            time_window_hours: Time window to schedule tasks within
            
        Returns:
            List of scheduled task slots
        """
        if not self.config["adaptive_scheduling"]:
            return []
        
        # Get all pending tasks
        pending_tasks = [task for task in self.tasks.values() if task.status == "pending"]
        
        # Sort by priority and dependencies
        sorted_tasks = self._sort_tasks_optimally(pending_tasks)
        
        # Generate schedule
        schedule = []
        current_time = datetime.now()
        
        for task in sorted_tasks:
            # Check if dependencies are satisfied
            if not self._dependencies_satisfied(task):
                continue
            
            # Find optimal time slot
            optimal_time = self._find_optimal_time_slot(task, current_time, time_window_hours)
            
            schedule_entry = {
                "task_id": task.id,
                "task_name": task.name,
                "scheduled_time": optimal_time,
                "estimated_duration": task.estimated_duration,
                "priority": task.priority,
                "confidence": self._calculate_scheduling_confidence(task, optimal_time)
            }
            
            schedule.append(schedule_entry)
            
            # Update current time for next task
            current_time = optimal_time + timedelta(minutes=task.estimated_duration)
            
            # Break if we exceed time window
            if (current_time - datetime.now()).total_seconds() > time_window_hours * 3600:
                break
        
        return schedule
    
    def update_context(self, context_updates: Dict[str, Any]):
        """
        Update the current context (location, activity, mood, etc.).
        
        Args:
            context_updates: Dictionary of context updates
        """
        if not hasattr(self, '_current_context'):
            self._current_context = {}
        
        self.active_contexts.update(context_updates)
        self._current_context.update(context_updates)
        
        # Record context change
        self._record_user_action("context_updated", {
            "context_updates": context_updates,
            "full_context": self.active_contexts.copy()
        })
        
        logger.debug(f"Context updated: {context_updates}")
        
        # Trigger re-evaluation if context-aware
        if self.config["context_awareness"]:
            self._reevaluate_suggestions()
    
    def learn_from_feedback(self, suggestion_id: str, feedback: str, rating: int = None):
        """
        Learn from user feedback on suggestions.
        
        Args:
            suggestion_id: ID of the suggestion
            feedback: User feedback text
            rating: Optional rating (1-5)
        """
        feedback_data = {
            "suggestion_id": suggestion_id,
            "feedback": feedback,
            "rating": rating,
            "timestamp": datetime.now()
        }
        
        self._record_user_action("suggestion_feedback", feedback_data)
        
        # Update suggestion engine with feedback
        self.suggestion_engine.learn_from_feedback(feedback_data)
        
        logger.info(f"Learned from feedback for suggestion: {suggestion_id}")
    
    def _generate_task_id(self, name: str) -> str:
        """Generate a unique task ID."""
        timestamp = str(int(time.time()))
        name_hash = hashlib.md5(name.encode()).hexdigest()[:8]
        return f"task_{timestamp}_{name_hash}"
    
    def _record_user_action(self, action_type: str, data: Dict[str, Any]):
        """Record user action for learning purposes."""
        action_record = {
            "timestamp": datetime.now(),
            "action_type": action_type,
            "data": data,
            "context": self.active_contexts.copy()
        }
        
        self.task_history.append(action_record)
        
        # Update behavior patterns
        if self.config["learning_enabled"]:
            self.pattern_detector.update_patterns(action_record, self.behavior_patterns)
    
    def _monitoring_loop(self):
        """Main monitoring loop for the task facilitator."""
        logger.info("Task facilitator monitoring started")
        
        while self.is_active:
            try:
                # Process task queue
                self._process_task_queue()
                
                # Check for overdue tasks
                self._check_overdue_tasks()
                
                # Update task priorities based on context
                if self.config["auto_prioritize"]:
                    self._auto_prioritize_tasks()
                
                # Generate proactive suggestions
                if self.config["proactive_suggestions"]:
                    self._generate_proactive_suggestions()
                
                # Save data periodically
                if int(time.time()) % 300 == 0:  # Every 5 minutes
                    self._save_user_data()
                
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(60)
    
    def _learning_loop(self):
        """Learning loop for pattern detection and adaptation."""
        logger.info("Task facilitator learning started")
        
        while self.is_active:
            try:
                # Update behavior patterns
                self.pattern_detector.detect_patterns(
                    list(self.task_history), 
                    self.behavior_patterns
                )
                
                # Update task predictions
                self.task_predictor.update_predictions(
                    self.completed_tasks,
                    self.behavior_patterns
                )
                
                # Update suggestion engine
                self.suggestion_engine.update_models(
                    self.task_history,
                    self.behavior_patterns
                )
                
                time.sleep(self.config["pattern_detection_interval"])
                
            except Exception as e:
                logger.error(f"Error in learning loop: {e}")
                time.sleep(300)  # Wait 5 minutes on error
    
    def _generate_contextual_suggestions(self, task: Task):
        """Generate suggestions based on the context of a new task."""
        # This would use the suggestion engine to generate contextual suggestions
        pass
    
    def _save_user_data(self):
        """Save user data to disk."""
        try:
            data = {
                "tasks": {k: self._task_to_dict(v) for k, v in self.tasks.items()},
                "completed_tasks": {k: self._task_to_dict(v) for k, v in self.completed_tasks.items()},
                "behavior_patterns": {k: self._pattern_to_dict(v) for k, v in self.behavior_patterns.items()},
                "user_preferences": self.user_preferences,
                "config": self.config
            }
            
            user_data_file = os.path.join(self.data_dir, f"{self.user_id}_data.json")
            with open(user_data_file, 'w') as f:
                json.dump(data, f, default=str, indent=2)
            
            # Save task history separately (binary for efficiency)
            history_file = os.path.join(self.data_dir, f"{self.user_id}_history.pkl")
            with open(history_file, 'wb') as f:
                pickle.dump(list(self.task_history), f)
            
            logger.debug("User data saved successfully")
            
        except Exception as e:
            logger.error(f"Failed to save user data: {e}")
    
    def _load_user_data(self):
        """Load user data from disk."""
        try:
            user_data_file = os.path.join(self.data_dir, f"{self.user_id}_data.json")
            if os.path.exists(user_data_file):
                with open(user_data_file, 'r') as f:
                    data = json.load(f)
                
                # Load tasks
                for task_id, task_data in data.get("tasks", {}).items():
                    self.tasks[task_id] = self._dict_to_task(task_data)
                
                # Load completed tasks
                for task_id, task_data in data.get("completed_tasks", {}).items():
                    self.completed_tasks[task_id] = self._dict_to_task(task_data)
                
                # Load behavior patterns
                for pattern_id, pattern_data in data.get("behavior_patterns", {}).items():
                    self.behavior_patterns[pattern_id] = self._dict_to_pattern(pattern_data)
                
                # Load preferences and config
                self.user_preferences = data.get("user_preferences", {})
                self.config.update(data.get("config", {}))
            
            # Load task history
            history_file = os.path.join(self.data_dir, f"{self.user_id}_history.pkl")
            if os.path.exists(history_file):
                with open(history_file, 'rb') as f:
                    history_list = pickle.load(f)
                    self.task_history.extend(history_list)
            
            logger.info(f"User data loaded: {len(self.tasks)} active tasks, {len(self.completed_tasks)} completed")
            
        except Exception as e:
            logger.error(f"Failed to load user data: {e}")
    
    def _task_to_dict(self, task: Task) -> Dict[str, Any]:
        """Convert Task object to dictionary."""
        return {
            "id": task.id,
            "name": task.name,
            "description": task.description,
            "priority": task.priority,
            "estimated_duration": task.estimated_duration,
            "dependencies": task.dependencies,
            "tags": task.tags,
            "context": task.context,
            "created_at": task.created_at.isoformat(),
            "status": task.status,
            "progress": task.progress
        }
    
    def _dict_to_task(self, data: Dict[str, Any]) -> Task:
        """Convert dictionary to Task object."""
        return Task(
            id=data["id"],
            name=data["name"],
            description=data["description"],
            priority=data["priority"],
            estimated_duration=data["estimated_duration"],
            dependencies=data["dependencies"],
            tags=data["tags"],
            context=data["context"],
            created_at=datetime.fromisoformat(data["created_at"]),
            status=data["status"],
            progress=data["progress"]
        )
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status of the task facilitator."""
        # Determine overall status
        if self.is_active:
            overall_status = "active"
        else:
            overall_status = "inactive"
        
        return {
            "status": overall_status,  # Add status field that tests expect
            "is_active": self.is_active,
            "user_id": self.user_id,
            "active_tasks": len(self.tasks),
            "completed_tasks": len(self.completed_tasks),
            "behavior_patterns": len(self.behavior_patterns),
            "task_history_size": len(self.task_history),
            "active_contexts": self.active_contexts,
            "config": self.config
        }
    
    def _learn_from_completion(self, task: Task, actual_duration: float):
        """Learn from task completion patterns."""
        # Update learning patterns
        if not hasattr(self, '_learning_data'):
            self._learning_data = {}
        
        task_type = getattr(task, 'category', 'general')
        if task_type not in self._learning_data:
            self._learning_data[task_type] = {
                'average_duration': [],
                'completion_patterns': [],
                'success_factors': []
            }
        
        self._learning_data[task_type]['average_duration'].append(actual_duration)
        self._learning_data[task_type]['completion_patterns'].append({
            'priority': task.priority,
            'tags': task.tags,
            'duration': actual_duration,
            'completed_at': datetime.utcnow().isoformat()
        })
        
        # Keep only recent learning data
        if len(self._learning_data[task_type]['average_duration']) > 100:
            self._learning_data[task_type]['average_duration'] = \
                self._learning_data[task_type]['average_duration'][-50:]
    
    def _reevaluate_suggestions(self):
        """Re-evaluate task suggestions based on context changes."""
        if not hasattr(self, '_cached_suggestions'):
            self._cached_suggestions = []
        
        # Mark cached suggestions as stale
        for suggestion in self._cached_suggestions:
            suggestion.confidence *= 0.9  # Reduce confidence due to context change
        
        # Remove very low confidence suggestions
        self._cached_suggestions = [
            s for s in self._cached_suggestions if s.confidence > 0.3
        ]
    
    def _generate_context_suggestions(self) -> List[TaskSuggestion]:
        """Generate suggestions based on current context."""
        suggestions = []
        
        current_context = getattr(self, '_current_context', {})
        
        # Suggest tasks based on location
        location = current_context.get('location', 'unknown')
        if location == 'office':
            suggestions.append(TaskSuggestion(
                suggestion_id=f"context_office_{len(suggestions)}",
                task_name="Review daily priorities",
                reasoning="Common office task at this time",
                confidence=0.7,
                priority=6,
                estimated_benefit=0.8
            ))
        elif location == 'home':
            suggestions.append(TaskSuggestion(
                suggestion_id=f"context_home_{len(suggestions)}",
                task_name="Personal development reading",
                reasoning="Good time for personal growth activities at home",
                confidence=0.6,
                priority=5,
                estimated_benefit=0.7
            ))
        
        # Suggest tasks based on activity
        activity = current_context.get('activity', 'unknown')
        if activity == 'testing':
            suggestions.append(TaskSuggestion(
                suggestion_id=f"context_testing_{len(suggestions)}",
                task_name="Document test results",
                reasoning="Natural follow-up to testing activities",
                confidence=0.8,
                priority=7,
                estimated_benefit=0.9
            ))
        
        return suggestions
    
    def _process_task_queue(self):
        """Process pending tasks in the queue."""
        # Simple implementation for now
        pass
    
    def _check_overdue_tasks(self):
        """Check for overdue tasks."""
        current_time = datetime.utcnow()
        overdue_tasks = []
        
        for task in self.tasks.values():
            # Handle tasks with or without due_date
            due_date = getattr(task, 'due_date', None)
            completed = getattr(task, 'completed', False)
            if due_date and due_date < current_time and not completed:
                overdue_tasks.append(task)
        
        return overdue_tasks
    
    def _generate_pattern_suggestions(self) -> List[TaskSuggestion]:
        """Generate suggestions based on behavioral patterns."""
        suggestions = []
        
        # Simple pattern-based suggestions
        if hasattr(self, 'task_history') and len(self.task_history) > 5:
            # Suggest similar tasks based on recent completions
            try:
                recent_tasks = list(self.task_history)[-5:]  # Convert to list first
                common_tags = set()
                for task_record in recent_tasks:
                    if isinstance(task_record, dict):
                        common_tags.update(task_record.get('tags', []))
                
                if 'research' in common_tags:
                    suggestions.append(TaskSuggestion(
                        suggestion_id=f"pattern_research_{len(suggestions)}",
                        task_name="Continue research activities",
                        reasoning="You've been doing research tasks recently",
                        confidence=0.6,
                        priority=6,
                        estimated_benefit=0.7
                    ))
                    
                if 'development' in common_tags:
                    suggestions.append(TaskSuggestion(
                        suggestion_id=f"pattern_dev_{len(suggestions)}",
                        task_name="Review development progress",
                        reasoning="Pattern shows focus on development work",
                        confidence=0.7,
                        priority=7,
                        estimated_benefit=0.8
                    ))
            except Exception as e:
                # Fallback: just add a generic suggestion
                suggestions.append(TaskSuggestion(
                    suggestion_id="pattern_generic_0",
                    task_name="Review recent work",
                    reasoning="Time to review and plan next steps",
                    confidence=0.5,
                    priority=5,
                    estimated_benefit=0.6
                ))
        
        return suggestions
    
    def _auto_prioritize_tasks(self):
        """Automatically prioritize tasks based on various factors."""
        # Simple auto-prioritization logic
        for task in self.tasks.values():
            # Increase priority for overdue tasks
            if hasattr(task, 'due_date') and task.due_date:
                current_time = datetime.utcnow()
                if task.due_date < current_time:
                    task.priority = min(10, task.priority + 2)
            
            # Increase priority for high-value tags
            if hasattr(task, 'tags'):
                high_priority_tags = ['urgent', 'important', 'deadline']
                if any(tag in task.tags for tag in high_priority_tags):
                    task.priority = min(10, task.priority + 1)
    
    def _generate_dependency_suggestions(self) -> List[TaskSuggestion]:
        """Generate suggestions based on task dependencies."""
        suggestions = []
        
        # Simple dependency-based suggestions
        suggestions.append(TaskSuggestion(
            suggestion_id="dependency_generic_0",
            task_name="Check task dependencies",
            reasoning="Review if any tasks are blocking others",
            confidence=0.4,
            priority=5,
            estimated_benefit=0.6
        ))
        
        return suggestions
    
    def _generate_proactive_suggestions(self):
        """Generate proactive task suggestions based on patterns and context."""
        suggestions = []
        
        # Add time-based suggestions
        current_hour = datetime.now().hour
        if 9 <= current_hour <= 11:  # Morning
            suggestions.append(TaskSuggestion(
                suggestion_id="proactive_morning_0",
                task_name="Plan daily priorities",
                reasoning="Good time for planning in the morning",
                confidence=0.6,
                priority=6,
                estimated_benefit=0.7
            ))
        elif 13 <= current_hour <= 15:  # Early afternoon
            suggestions.append(TaskSuggestion(
                suggestion_id="proactive_afternoon_0",
                task_name="Review progress and adjust plans",
                reasoning="Mid-day check-in time",
                confidence=0.5,
                priority=5,
                estimated_benefit=0.6
            ))
        
        return suggestions
    
    def _generate_time_based_suggestions(self) -> List[TaskSuggestion]:
        """Generate time-based task suggestions."""
        return self._generate_proactive_suggestions()  # Same logic
    
    def _check_dependent_tasks(self, task_id: str):
        """Check and update dependent tasks when a task is completed."""
        # Simple implementation - just log for now
        # In a real system, this would check task dependencies and unblock waiting tasks
        pass


# Helper classes for pattern detection, prediction, and suggestions
class PatternDetector:
    """Detects patterns in user behavior."""
    
    def update_patterns(self, action_record: Dict[str, Any], patterns: Dict[str, UserBehaviorPattern]):
        """Update patterns based on new action."""
        # Simplified pattern detection - would be more sophisticated in real implementation
        pass
    
    def detect_patterns(self, history: List[Dict[str, Any]], patterns: Dict[str, UserBehaviorPattern]):
        """Detect patterns from history."""
        # Analyze history for patterns
        pass

class TaskPredictor:
    """Predicts task completion times and success rates."""
    
    def update_predictions(self, completed_tasks: Dict[str, Task], patterns: Dict[str, UserBehaviorPattern]):
        """Update prediction models."""
        pass

class SuggestionEngine:
    """Generates intelligent task suggestions."""
    
    def update_models(self, history: List[Dict[str, Any]], patterns: Dict[str, UserBehaviorPattern]):
        """Update suggestion models."""
        pass
    
    def learn_from_feedback(self, feedback_data: Dict[str, Any]):
        """Learn from user feedback."""
        pass
    
    def _learn_from_completion(self, task: Task, actual_duration: float):
        """Learn from task completion patterns."""
        # Update learning patterns
        if not hasattr(self, '_learning_data'):
            self._learning_data = {}
        
        task_type = getattr(task, 'category', 'general')
        if task_type not in self._learning_data:
            self._learning_data[task_type] = {
                'average_duration': [],
                'completion_patterns': [],
                'success_factors': []
            }
        
        self._learning_data[task_type]['average_duration'].append(actual_duration)
        self._learning_data[task_type]['completion_patterns'].append({
            'priority': task.priority,
            'tags': task.tags,
            'duration': actual_duration,
            'completed_at': datetime.utcnow().isoformat()
        })
        
        # Keep only recent learning data
        if len(self._learning_data[task_type]['average_duration']) > 100:
            self._learning_data[task_type]['average_duration'] = \
                self._learning_data[task_type]['average_duration'][-50:]
    
    def _reevaluate_suggestions(self):
        """Re-evaluate task suggestions based on context changes."""
        if not hasattr(self, '_cached_suggestions'):
            self._cached_suggestions = []
        
        # Mark cached suggestions as stale
        for suggestion in self._cached_suggestions:
            suggestion.confidence *= 0.9  # Reduce confidence due to context change
        
        # Remove very low confidence suggestions
        self._cached_suggestions = [
            s for s in self._cached_suggestions if s.confidence > 0.3
        ]
    
    def _generate_context_suggestions(self) -> List[TaskSuggestion]:
        """Generate suggestions based on current context."""
        suggestions = []
        
        current_context = getattr(self, '_current_context', {})
        
        # Suggest tasks based on location
        location = current_context.get('location', 'unknown')
        if location == 'office':
            suggestions.append(TaskSuggestion(
                suggestion_id=f"context_office_{len(suggestions)}",
                task_name="Review daily priorities",
                reasoning="Common office task at this time",
                confidence=0.7,
                priority=6,
                estimated_benefit=0.8
            ))
        elif location == 'home':
            suggestions.append(TaskSuggestion(
                suggestion_id=f"context_home_{len(suggestions)}",
                task_name="Personal development reading",
                reasoning="Good time for personal growth activities at home",
                confidence=0.6,
                priority=5,
                estimated_benefit=0.7
            ))
        
        # Suggest tasks based on activity
        activity = current_context.get('activity', 'unknown')
        if activity == 'testing':
            suggestions.append(TaskSuggestion(
                suggestion_id=f"context_testing_{len(suggestions)}",
                task_name="Document test results",
                reasoning="Natural follow-up to testing activities",
                confidence=0.8,
                priority=7,
                estimated_benefit=0.9
            ))
        
        return suggestions


def main():
    """Test the Adaptive Task Facilitator."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Adaptive Task Facilitator")
    parser.add_argument("--user-id", default="test_user", help="User ID")
    parser.add_argument("--add-task", help="Add a test task")
    parser.add_argument("--status", action="store_true", help="Show status")
    parser.add_argument("--suggestions", action="store_true", help="Get task suggestions")
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create task facilitator
    facilitator = AdaptiveTaskFacilitator(user_id=args.user_id)
    facilitator.start()
    
    try:
        if args.add_task:
            task_id = facilitator.add_task(
                name=args.add_task,
                description=f"Test task: {args.add_task}",
                priority=7,
                tags=["test", "demo"]
            )
            print(f"Added task: {task_id}")
            
        elif args.status:
            status = facilitator.get_status()
            print(json.dumps(status, indent=2))
            
        elif args.suggestions:
            suggestions = facilitator.get_task_suggestions()
            print("Task suggestions:")
            for suggestion in suggestions:
                print(f"- {suggestion.task_name}: {suggestion.reasoning} (confidence: {suggestion.confidence:.2f})")
        
        else:
            parser.print_help()
            
        # Keep running for a bit to test monitoring
        if args.add_task:
            print("Running for 30 seconds to test monitoring...")
            time.sleep(30)
    
    finally:
        facilitator.stop()


if __name__ == "__main__":
    main()