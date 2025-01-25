import time
import logging
import boto3
from botocore.exceptions import ClientError
import textwrap
import json
# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class LLMUtils:
    def __init__(self, region_name):
        """
        Initialize the LLMUtils with Bedrock clients.

        Args:
            region_name (str): The AWS region to connect to.
        """
        self.region_name = region_name
        self.bedrock_agent = boto3.client(service_name='bedrock-agent', region_name=region_name)
        self.bedrock_agent_runtime = boto3.client(service_name='bedrock-agent-runtime', region_name=region_name)

    def wait_for_agent_alias_status(self, agent_id: str, agent_alias_id: str, target_status: str, max_retries: int = 3, wait_time: int = 5):
        """
        Waits for the agent alias to reach the target status.

        :param agent_id: The unique identifier of the agent.
        :param agent_alias_id: The unique identifier of the agent alias.
        :param target_status: The desired status to wait for.
        :param max_retries: Maximum number of retries.
        :param wait_time: Time to wait between retries in seconds.
        :raises RuntimeError: If the alias does not reach the target status after max_retries.
        """
        retries = 0

        logger.info(f"Waiting for alias '{agent_alias_id}' to reach status '{target_status}'...")

        while retries < max_retries:
            try:
                response = self.bedrock_agent.get_agent_alias(agentId=agent_id, agentAliasId=agent_alias_id)
                current_status = response['agentAlias']['agentAliasStatus']

                if current_status == target_status:
                    logger.info(f"Alias '{agent_alias_id}' reached status '{target_status}'.")
                    return
                elif current_status in ["FAILED", "STOPPED"]:
                    logger.warning(f"Alias '{agent_alias_id}' is in '{current_status}'.")
                    break
                else:
                    logger.info(f"Alias '{agent_alias_id}' is in '{current_status}', retrying in {wait_time} seconds...")

            except ClientError as e:
                logger.error(f"Error fetching status for alias '{agent_alias_id}': {e}")
                break

            time.sleep(wait_time)
            retries += 1

        logger.error(f"Failed to reach status '{target_status}' for alias '{agent_alias_id}' after {max_retries} attempts.")
        raise RuntimeError(f"Alias '{agent_alias_id}' did not reach status '{target_status}' after {max_retries} attempts.")



    def wait_for_agent_status(self, agentId: str, targetStatus: str, agentName: str, agentResourceRoleArn: str, foundationModel: str):
        """
        Wait for a Bedrock agent to reach the specified status, moving it to the desired status if necessary.

        Args:
            agentId (str): The ID of the Bedrock agent.
            targetStatus (str): The desired agent status.
            agentName (str): The name of the agent.
            agentResourceRoleArn (str): The IAM role ARN associated with the agent.
            foundationModel (str): The foundation model used by the agent.

        Returns:
            None
        """
        logger.info(f"Ensuring agent '{agentId}' transitions to status '{targetStatus}'...")

        retries = 0
        max_retries = 3
        wait_time = 5  # Seconds to wait between retries

        while retries < max_retries:
            # Fetch the current status of the agent
            response = self.bedrock_agent.get_agent(agentId=agentId)
            current_status = response['agent']['agentStatus']
            logger.info(f"Current agent status: {current_status}")

            # If the agent is already in the target status, exit successfully
            if current_status == targetStatus:
                logger.info(f"Agent '{agentId}' reached status '{targetStatus}'.")
                return

            # Handle NOT_PREPARED by triggering a re-preparation
            elif current_status == "NOT_PREPARED":
                logger.warning(f"Agent '{agentId}' is in 'NOT_PREPARED' state. Attempting to prepare...")
                try:
                    response = self.bedrock_agent.prepare_agent(agentId=agentId)
                    logger.info(f"Agent preparation initiated: {response}")
                    logger.info(f"Preparation action triggered for agent '{agentId}'.")
                except Exception as e:
                    logger.error(f"Error while triggering preparation for agent '{agentId}': {str(e)}")
                    raise

            # Handle intermediate states
            elif current_status in ["CREATING", "PREPARING", "UPDATING"]:
                logger.info(f"Agent '{agentId}' is in progress ('{current_status}'). Waiting to transition...")

            # Handle terminal or unexpected states
            elif current_status in ["FAILED", "DELETING"]:
                logger.warning(f"Agent '{agentId}' is in terminal state '{current_status}'. Attempting recovery...")
                try:
                    response = self.bedrock_agent.prepare_agent(agentId=agentId)
                    logger.info(f"Agent preparation initiated: {response}")
                    logger.info(f"Recovery action triggered for agent '{agentId}'.")
                except Exception as e:
                    logger.error(f"Error while recovering agent '{agentId}': {str(e)}")
                    raise

            else:
                logger.error(f"Unhandled agent status '{current_status}'. Manual intervention may be required.")
                raise RuntimeError(f"Cannot transition agent '{agentId}' to '{targetStatus}' from status '{current_status}'.")

            # Wait before retrying
            time.sleep(wait_time)
            retries += 1

        # If retries are exceeded, log and raise an error
        logger.error(f"Failed to transition agent '{agentId}' to status '{targetStatus}' after {max_retries} attempts.")
        raise RuntimeError(f"Agent '{agentId}' did not reach status '{targetStatus}' after {max_retries} attempts.")




    def is_ready(self, agentId: str, agentAliasId: str) -> bool:
        """
        Check if both the agent and its alias are in the 'PREPARED' state.

        Args:
            agentId (str): The ID of the Bedrock agent.
            agentAliasId (str): The ID of the agent alias.

        Returns:
            bool: True if both agent and alias are prepared, False otherwise.
        """
        agent_status = self.bedrock_agent.get_agent(agentId=agentId)['agent']['agentStatus']
        alias_status = self.bedrock_agent.get_agent_alias(agentId=agentId, agentAliasId=agentAliasId)['agentAlias']['agentAliasStatus']
        
        logger.info(f"Agent status: {agent_status}, Alias status: {alias_status}")
        return agent_status == 'PREPARED' and alias_status == 'PREPARED'

    def invoke_agent(self, agentId: str, agentAliasId: str, inputText: str, sessionId: str, enableTrace: bool = False, endSession: bool = False, width: int = 70):
        """
        Invoke the Bedrock agent and process the EventStream response.

        Args:
            agentId (str): The ID of the agent.
            agentAliasId (str): The ID of the agent alias.
            inputText (str): The user input text.
            sessionId (str): The session ID.
            enableTrace (bool): Whether to enable trace details.
            endSession (bool): Whether to end the session after this invocation.
            width (int): The width for text wrapping in logs.

        Returns:
            dict: A dictionary containing the agent's response and session details.
        """
        bedrock_agent_runtime = boto3.client(service_name='bedrock-agent-runtime', region_name=self.region_name)
        
        try:
            logger.info(f"Invoking agent '{agentId}' with alias '{agentAliasId}' for session '{sessionId}'...")
            response = bedrock_agent_runtime.invoke_agent(
                agentId=agentId,
                agentAliasId=agentAliasId,
                sessionId=sessionId,
                inputText=inputText,
                endSession=endSession,
                enableTrace=enableTrace
            )

            event_stream = response["completion"]
            agent_response = ""

            logger.info(f"User: {textwrap.fill(inputText, width=width)}")
            logger.info("Agent response in progress...")

            for event in event_stream:
                if 'chunk' in event:
                    # Process chunks of response text
                    chunk_text = event['chunk'].get('bytes', b'').decode('utf-8')
                    logger.debug(f"Received chunk: {chunk_text}")
                    agent_response += chunk_text

                elif 'trace' in event and enableTrace:
                    # Process trace details if enabled
                    trace = event['trace']
                    if 'trace' in trace:
                        trace_details = trace['trace']

                        if 'orchestrationTrace' in trace_details:
                            orch_trace = trace_details['orchestrationTrace']

                            if 'invocationInput' in orch_trace:
                                inv_input = orch_trace['invocationInput']
                                logger.info(f"Invocation Input: {json.dumps(inv_input, indent=2)}")

                            if 'rationale' in orch_trace:
                                thought = orch_trace['rationale']['text']
                                logger.info(f"Agent's thought process: {thought}")

                            if 'observation' in orch_trace:
                                obs = orch_trace['observation']
                                logger.info(f"Observation: {json.dumps(obs, indent=2)}")

                        if 'guardrailTrace' in trace_details:
                            guard_trace = trace_details['guardrailTrace']
                            logger.info(f"Guardrail Trace: {json.dumps(guard_trace, indent=2)}")

            logger.info(f"Agent response complete. Session ID: {response.get('sessionId')}")

            return {
                "agentResponse": agent_response,
                "sessionId": response.get('sessionId')
            }
        except boto3.exceptions.Boto3Error as e:
            if "accessDeniedException" in str(e):
                logger.error("Access denied. Please check the IAM permissions for Bedrock.")
                raise RuntimeError("Access denied. Ensure the Lambda execution role has the necessary permissions.")
        except Exception as e:
            logger.error(f"Error invoking agent: {str(e)}")
            raise RuntimeError(f"Failed to invoke agent: {str(e)}")

    
    def delete_agent(self, agentId: str):
        """
        Delete an agent and its associated aliases.

        Args:
            agentId (str): The ID of the Bedrock agent to delete.

        Returns:
            None
        """
        try:
            logger.info(f"Fetching aliases for agent '{agentId}'...")
            # List all aliases for the agent
            alias_response = self.bedrock_agent.list_agent_aliases(agentId=agentId)
            aliases = alias_response.get('agentAliasSummaries', [])

            # Delete each alias
            for alias in aliases:
                alias_id = alias['agentAliasId']
                alias_name = alias['agentAliasName']
                logger.info(f"Deleting alias '{alias_name}' (ID: {alias_id})...")
                try:
                    self.bedrock_agent.delete_agent_alias(agentId=agentId, agentAliasId=alias_id)
                    logger.info(f"Alias '{alias_name}' deletion initiated. Waiting for deletion...")
                    while True:
                        try:
                            self.bedrock_agent.get_agent_alias(agentId=agentId, agentAliasId=alias_id)
                            logger.info(f"Alias '{alias_name}' still exists. Retrying in 5 seconds...")
                            time.sleep(5)
                        except self.bedrock_agent.exceptions.ResourceNotFoundException:
                            logger.info(f"Alias '{alias_name}' successfully deleted.")
                            break
                except Exception as e:
                    logger.error(f"Error deleting alias '{alias_name}': {str(e)}")

            # Delete the agent
            logger.info(f"Deleting agent '{agentId}'...")
            self.bedrock_agent.delete_agent(agentId=agentId)
            logger.info(f"Agent '{agentId}' deletion initiated. Waiting for deletion...")
            while True:
                try:
                    self.bedrock_agent.get_agent(agentId=agentId)
                    logger.info(f"Agent '{agentId}' still exists. Retrying in 3 seconds...")
                    time.sleep(3)
                except self.bedrock_agent.exceptions.ResourceNotFoundException:
                    logger.info(f"Agent '{agentId}' successfully deleted.")
                    break

        except Exception as e:
            logger.error(f"Error deleting agent '{agentId}': {str(e)}")

