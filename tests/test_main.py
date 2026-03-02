import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from system_architect_advisor_agent.main import handler


@pytest.mark.asyncio
async def test_handler_returns_response():
    """Test that handler accepts messages and returns a response."""
    messages = [{"role": "user", "content": "Hello, how are you?"}]

    # Mock run_architect_flow function to return a mock response
    mock_response = "Test architectural analysis response"

    # Mock _initialized to skip initialization and run_architect_flow to return our mock
    with patch("system_architect_advisor_agent.main._initialized", True), \
         patch("system_architect_advisor_agent.main.run_architect_flow", new_callable=AsyncMock, return_value=mock_response):
        result = await handler(messages)

    # Verify we get a result back
    assert result is not None
    assert result == "Test architectural analysis response"


@pytest.mark.asyncio
async def test_handler_with_multiple_messages():
    """Test that handler processes multiple messages correctly."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What's the weather?"},
    ]

    mock_response = "Multiple messages test response"

    with patch("system_architect_advisor_agent.main._initialized", True), \
         patch("system_architect_advisor_agent.main.run_architect_flow", new_callable=AsyncMock, return_value=mock_response) as mock_run:
        result = await handler(messages)

    # Verify run_architect_flow was called
    mock_run.assert_called_once_with(messages)
    assert result is not None
    assert result == "Multiple messages test response"


@pytest.mark.asyncio
async def test_handler_initialization():
    """Test that handler initializes on first call."""
    _messages = [{"role": "user", "content": "Test"}]

    mock_response = "Initialization test response"

    # Start with _initialized as False to test initialization path
    with patch("system_architect_advisor_agent.main._initialized", False), \
         patch("system_architect_advisor_agent.main.initialize_agent", new_callable=AsyncMock) as mock_init, \
         patch("system_architect_advisor_agent.main.run_architect_flow", new_callable=AsyncMock, return_value=mock_response):
        result = await handler(_messages)

    # Verify initialization was called
    mock_init.assert_called_once()
    assert result is not None
    assert result == "Initialization test response"


@pytest.mark.asyncio
async def test_handler_race_condition_prevention():
    """Test that handler prevents race conditions with initialization lock."""
    messages = [{"role": "user", "content": "Test"}]

    mock_response = "Race condition test response"

    # Test with multiple concurrent calls
    with (
        patch("system_architect_advisor_agent.main._initialized", False),
        patch("system_architect_advisor_agent.main.initialize_agent", new_callable=AsyncMock) as mock_init,
        patch("system_architect_advisor_agent.main.run_architect_flow", new_callable=AsyncMock, return_value=mock_response),
        patch("system_architect_advisor_agent.main._init_lock", new_callable=MagicMock()) as mock_lock,
    ):
        # Configure the lock to work as an async context manager
        mock_lock_instance = MagicMock()
        mock_lock_instance.__aenter__ = AsyncMock(return_value=None)
        mock_lock_instance.__aexit__ = AsyncMock(return_value=None)
        mock_lock.return_value = mock_lock_instance

        # Call handler twice to ensure lock is used
        await handler(messages)
        await handler(messages)

        # Verify initialize_agent was called only once (due to lock)
        mock_init.assert_called_once()


@pytest.mark.asyncio
async def test_handler_with_architectural_query():
    """Test that handler can process an architectural query."""
    messages = [
        {
            "role": "user",
            "content": "Design a microservices architecture for an e-commerce platform",
        }
    ]

    mock_response = "# E-commerce Microservices Architecture\n\n## System Design\n..."

    with (
        patch("system_architect_advisor_agent.main._initialized", True),
        patch("system_architect_advisor_agent.main.run_architect_flow", new_callable=AsyncMock, return_value=mock_response),
    ):
        result = await handler(messages)

    assert result is not None
    assert "# E-commerce Microservices Architecture" in result
    assert "## System Design" in result
