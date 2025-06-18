from __future__ import annotations as _annotations

import os
from typing import overload

from pydantic_ai.exceptions import UserError
from pydantic_ai.providers import Provider

try:
    from aws_sdk_bedrock_runtime.client import BedrockRuntimeClient
    from aws_sdk_bedrock_runtime.config import Config, HTTPRequestConfiguration
    from smithy_aws_core.credentials_resolvers import StaticCredentialsResolver
    from smithy_aws_core.identity import AWSCredentialsIdentity

    # from botocore.client import BaseClient
    # from botocore.config import Config
    # from botocore.exceptions import NoRegionError
except ImportError as _import_error:
    raise ImportError(
        'Please install the `aws_sdk_bedrock_runtime` package to use the experimental Bedrock provider, '
        'you can use the `bedrock-experimental` optional group — `pip install "pydantic-ai-slim[bedrock-experimental]"`'
        '\n\n'
        'NOTE: `aws_sdk_bedrock_runtime` is under active development and may break in future releases. '
        'This provider should be considered experimental until the official 1.0 release.'
    ) from _import_error


class ExperimentalBedrockProvider(Provider[BedrockRuntimeClient]):
    """Experimental Provider for AWS Bedrock."""

    @property
    def name(self) -> str:
        return 'bedrock-experimental'

    @property
    def base_url(self) -> str:
        raise NotImplementedError('base_url')
        # return self._client._config.

    @property
    def client(self) -> BedrockRuntimeClient:
        return self._client

    @overload
    def __init__(self, *, bedrock_client: BedrockRuntimeClient) -> None: ...

    @overload
    def __init__(
        self,
        *,
        region_name: str | None = None,
        # aws_access_key_id: str | None = None,
        # aws_secret_access_key: str | None = None,
        # aws_session_token: str | None = None,
        aws_read_timeout: float | None = None,
        # aws_connect_timeout: float | None = None,
    ) -> None: ...

    def __init__(
        self,
        *,
        bedrock_client: BedrockRuntimeClient | None = None,
        region_name: str | None = None,
        aws_access_key_id: str | None = None,
        aws_secret_access_key: str | None = None,
        aws_session_token: str | None = None,
        aws_read_timeout: float | None = None,
        # aws_connect_timeout: float | None = None,
    ) -> None:
        """Initialize the Bedrock provider.

        Args:
            bedrock_client: A aws_sdk_bedrock_runtime client for Bedrock Runtime. If provided, other arguments are ignored.
            region_name: The AWS region name.
            aws_access_key_id: The AWS access key ID.
            aws_secret_access_key: The AWS secret access key.
            aws_session_token: The AWS session token.
            aws_read_timeout: The read timeout for Bedrock client.
            aws_connect_timeout: The connect timeout for Bedrock client.
        """
        if bedrock_client is not None:
            self._client = bedrock_client
        else:
            try:
                # connect_timeout = aws_connect_timeout or float(os.getenv('AWS_CONNECT_TIMEOUT', 60))
                read_timeout = aws_read_timeout or float(os.getenv('AWS_READ_TIMEOUT', 300))

                access_key_id = aws_access_key_id or os.getenv('AWS_ACCESS_KEY_ID')
                assert access_key_id is not None, 'aws_access_key_id is required'

                secret_access_key = aws_secret_access_key or os.getenv('AWS_SECRET_ACCESS_KEY')
                assert secret_access_key is not None, 'aws_secret_access_key is required'

                session_token = aws_session_token or os.getenv('AWS_SESSION_TOKEN')  # Optional, can be None

                region = region_name or os.getenv('AWS_REGION') or os.getenv('AWS_DEFAULT_REGION')

                aws_credentials_identity_resolver = StaticCredentialsResolver(
                    credentials=AWSCredentialsIdentity(
                        access_key_id=access_key_id,
                        secret_access_key=secret_access_key,
                        session_token=session_token,
                    )
                )

                config = Config(
                    region=region,
                    aws_credentials_identity_resolver=aws_credentials_identity_resolver,
                    http_request_config=HTTPRequestConfiguration(read_timeout=read_timeout),
                )

                self._client = BedrockRuntimeClient(
                    config=config,
                )

            except Exception as exc:  # pragma: no cover
                raise UserError('Unknown error') from exc
