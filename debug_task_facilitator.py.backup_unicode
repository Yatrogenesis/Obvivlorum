"""Debug específico para TaskFacilitator"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from adaptive_task_facilitator import AdaptiveTaskFacilitator

def test_methods():
    print("Testing TaskFacilitator methods...")
    
    facilitator = AdaptiveTaskFacilitator("debug_user")
    facilitator.start()
    
    # Test que los métodos existen
    methods_to_test = [
        '_reevaluate_suggestions',
        '_learn_from_completion', 
        '_generate_context_suggestions'
    ]
    
    for method_name in methods_to_test:
        if hasattr(facilitator, method_name):
            print(f"OK - Method {method_name} exists")
        else:
            print(f"ERROR - Method {method_name} NOT FOUND")
    
    # Test add task
    task_id = facilitator.add_task("Debug task", description="Test task", priority=5)
    print(f"OK - Task added: {task_id}")
    
    # Test update context
    try:
        facilitator.update_context({"location": "office", "activity": "debugging"})
        print("OK - Context updated")
    except Exception as e:
        print(f"ERROR - Context update failed: {e}")
    
    # Test suggestions
    try:
        suggestions = facilitator.get_task_suggestions(max_suggestions=2)
        print(f"OK - Got {len(suggestions)} suggestions")
    except Exception as e:
        print(f"ERROR - Suggestions failed: {e}")
    
    facilitator.stop()

if __name__ == "__main__":
    test_methods()