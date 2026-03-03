
import asyncio
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner

# Import the agent from the main file
from weather_agent_stateful import root_agent

# Configuration
APP_NAME = "weather_tutorial_stateful"
USER_ID = "test_user"
SESSION_ID = "session_001"


async def call_agent_async(query: str, runner: Runner, user_id: str, session_id: str):
    """Helper function to send a query and display the response."""
    print(f"\n{'='*60}")
    print(f"USER: {query}")
    print(f"{'='*60}")
    
    response = await runner.run(
        user_id=user_id,
        session_id=session_id,
        new_message=query
    )
    
    print(f"AGENT: {response.message_text}")
    print(f"{'='*60}")


async def run_stateful_demo():
    """Demonstrates the stateful weather agent with automatic unit detection."""
    
    # 1. Initialize session with default Celsius preference
    print("\n🚀 Initializing Stateful Weather Agent Demo")
    print("="*60)
    
    session_service = InMemorySessionService()
    
    # Create session with initial state
    initial_state = {
        "user_preference_temperature_unit": "Celsius"
    }
    
    session = await session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=SESSION_ID,
        state=initial_state
    )
    
    print(f"✅ Session created with initial temperature unit: Celsius")
    print(f"   App: {APP_NAME}")
    print(f"   User: {USER_ID}")
    print(f"   Session: {SESSION_ID}")
    
    # 2. Create runner
    runner = Runner(
        agent=root_agent,
        app_name=APP_NAME,
        session_service=session_service
    )
    
    print(f"✅ Runner created for agent '{root_agent.name}'")
    
    # 3. Demo conversation flow
    print("\n" + "="*60)
    print("DEMO: Automatic Temperature Unit Detection")
    print("="*60)
    
    # Test 1: Initial weather check (should use Celsius)
    await call_agent_async(
        query="What's the weather in London?",
        runner=runner,
        user_id=USER_ID,
        session_id=SESSION_ID
    )
    
    # Test 2: User wants to switch to Fahrenheit (agent should detect and use tool)
    await call_agent_async(
        query="Can you switch to Fahrenheit please?",
        runner=runner,
        user_id=USER_ID,
        session_id=SESSION_ID
    )
    
    # Test 3: Check weather again (should now use Fahrenheit)
    await call_agent_async(
        query="Tell me the weather in New York",
        runner=runner,
        user_id=USER_ID,
        session_id=SESSION_ID
    )
    
    # Test 4: Another way to change units (agent should detect)
    await call_agent_async(
        query="I prefer Celsius",
        runner=runner,
        user_id=USER_ID,
        session_id=SESSION_ID
    )
    
    # Test 5: Verify it's back to Celsius
    await call_agent_async(
        query="What about Tokyo?",
        runner=runner,
        user_id=USER_ID,
        session_id=SESSION_ID
    )
    
    # Test 6: Test greeting delegation (should still work)
    await call_agent_async(
        query="Hello!",
        runner=runner,
        user_id=USER_ID,
        session_id=SESSION_ID
    )
    
    # Test 7: Test farewell delegation
    await call_agent_async(
        query="Thanks, goodbye!",
        runner=runner,
        user_id=USER_ID,
        session_id=SESSION_ID
    )
    
    # 4. Inspect final session state
    print("\n" + "="*60)
    print("FINAL SESSION STATE INSPECTION")
    print("="*60)
    
    final_session = await session_service.get_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=SESSION_ID
    )
    
    if final_session:
        print(f"✅ Temperature Unit Preference: {final_session.state.get('user_preference_temperature_unit', 'Not Set')}")
        print(f"✅ Last City Checked: {final_session.state.get('last_city_checked', 'Not Set')}")
        print(f"✅ Last Response (via output_key): {final_session.state.get('last_response', 'Not Set')[:100]}...")
        print(f"\n📋 Full State Dictionary:")
        for key, value in final_session.state.items():
            if key != 'last_response':  # Skip long response text
                print(f"   - {key}: {value}")
    else:
        print("❌ Could not retrieve final session state")
    
    print("\n" + "="*60)
    print("✅ Demo completed successfully!")
    print("="*60)


# Run the demo
if __name__ == "__main__":
    print("\n" + "🌡️ "*20)
    print("Stateful Weather Agent Demo - Automatic Unit Detection")
    print("🌡️ "*20)
    
    asyncio.run(run_stateful_demo())
