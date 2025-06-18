import pytest

from ..conftest import TestEnv, try_import

with try_import() as imports_successful:
    from aws_sdk_bedrock_runtime.models import ContentBlockText, ConverseInput, Message

    from pydantic_ai.providers.bedrock_experimental import ExperimentalBedrockProvider


pytestmark = pytest.mark.skipif(not imports_successful(), reason='bedrock-experimental not installed')


def test_bedrock_experimental_provider(env: TestEnv):
    env.set('AWS_DEFAULT_REGION', 'us-east-1')
    provider = ExperimentalBedrockProvider()
    assert isinstance(provider, ExperimentalBedrockProvider)
    assert provider.name == 'bedrock-experimental'
    # assert provider.base_url == 'https://bedrock-runtime.us-east-1.amazonaws.com' # TODO


def test_bedrock_experimental_provider_timeout(env: TestEnv):
    env.set('AWS_DEFAULT_REGION', 'us-east-1')
    env.set('AWS_READ_TIMEOUT', '1')
    # env.set('AWS_CONNECT_TIMEOUT', '1')
    provider = ExperimentalBedrockProvider()
    assert isinstance(provider, ExperimentalBedrockProvider)
    assert provider.name == 'bedrock-experimental'

    config = provider.client._config  # type: ignore
    assert config.http_request_config is not None
    assert config.http_request_config.read_timeout == 1


@pytest.mark.anyio
async def test_bedrock_experimental_can_invoke(env: TestEnv):
    provider = ExperimentalBedrockProvider(region_name='us-east-2')

    resp = await provider.client.converse(
        ConverseInput(
            model_id='us.anthropic.claude-3-7-sonnet-20250219-v1:0',
            messages=[
                Message(
                    role='user',
                    content=[
                        ContentBlockText('Hello. How are you?'),
                    ],
                )
            ],
        )
    )
