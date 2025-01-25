import boto3
import uuid
import json
import logging
import os
from utils.llmutil import LLMUtils  # Import the LLMUtils helper class

# Initialize logging
log_level_name = "INFO"
logger = logging.getLogger()
logger.setLevel(getattr(logging, log_level_name.upper(), logging.INFO))
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)')
ch = logging.StreamHandler()
ch.setFormatter(formatter)
logger.addHandler(ch)

# Load configuration from llm.config file
config_path = os.path.join(os.path.dirname(__file__), "llmconfig.json")
config = {}
with open(config_path, "r") as f:
    config = json.load(f)
    logger.info(f"agents config:{config}")
# Load instruction from system_instructions.txt
instruction_file_path = os.path.join(os.path.dirname(__file__), "system_instructions.txt")
instruction = "You are a helpful AI."  # Default instruction
try:
    with open(instruction_file_path, "r") as f:
        instruction = f.read().strip()  # Remove any leading or trailing whitespace
except FileNotFoundError:
    logger.warning(f"{instruction_file_path} not found. Using default instruction: '{instruction}'")
region_name = config.get("region_name", "us-east-1")
agent_name = config.get("agentName", "Doc3LLM-Agent")
agent_alias= config.get("agent_alias", "Doc3LLM-Agent-Alias")
foundation_model = config.get("foundationModel", "anthropic.claude-3-haiku-20240307-v1:0")
agent_role_arn = config.get("agentResourceRoleArn", "")
agentTrace = config.get("agentTrace", "false")
deleteAgent= config.get("deleteAgent", "false")

# Global variables to store the agent and alias information
agent_id = None
agent_alias_id = None

# Initialize LLMUtils instance
llm_utils = LLMUtils(region_name=region_name)



def bootstrap():
    """
    Initialize the Bedrock agent and its alias during Lambda bootstrap.
    """
    global agent_id, agent_alias_id

    try:
        logger.info("Checking for existing Bedrock agent and alias...")

        # Check if any existing agents exist
        existing_agents = llm_utils.bedrock_agent.list_agents().get('agentSummaries', [])
        logger.info(f"Existing agents: {existing_agents}")

        # Check if the specific agent exists
        target_agent = next((a for a in existing_agents if a['agentName'] == agent_name), None)

        if target_agent:
            logger.info(f"Found existing agent: {target_agent['agentName']} with ID: {target_agent['agentId']}, Status: {target_agent['agentStatus']}")

            if str(deleteAgent).lower() == "true":
                # Delete the specified agent and its aliases if deleteAgent is true
                logger.info(f"Deleting agent '{target_agent['agentName']}' with ID '{target_agent['agentId']}'...")
                llm_utils.delete_agent(agentId=target_agent['agentId'])
                logger.info(f"Agent '{target_agent['agentName']}' deleted successfully.")
                target_agent = None  # Clear target_agent after deletion
        else:
            logger.info(f"No existing agent found with name: {agent_name}")

        if not target_agent:
            # Create the agent if it doesn't exist
            logger.info("Creating a new Bedrock agent...")
            create_agent_response = llm_utils.bedrock_agent.create_agent(
                agentName=agent_name,
                foundationModel=foundation_model,
                instruction=instruction,
                agentResourceRoleArn=agent_role_arn
            )
            agent_id = create_agent_response['agent']['agentId']
            logger.info(f"Agent created with ID: {agent_id}")

            # Wait for the agent to be prepared
            llm_utils.wait_for_agent_status(
                agentId=agent_id,
                targetStatus="PREPARED",
                agentName=agent_name,
                agentResourceRoleArn=agent_role_arn,
                foundationModel=foundation_model
            )
        else:
            # Use the existing agent
            agent_id = target_agent['agentId']
            if target_agent['agentStatus'] != "PREPARED":
                logger.info(f"Agent '{agent_name}' is not prepared. Waiting for it to be prepared...")
                llm_utils.wait_for_agent_status(
                    agentId=agent_id,
                    targetStatus="PREPARED",
                    agentName=agent_name,
                    agentResourceRoleArn=agent_role_arn,
                    foundationModel=foundation_model
                )

        # Check if the specific alias exists
        existing_aliases = llm_utils.bedrock_agent.list_agent_aliases(agentId=agent_id).get('agentAliasSummaries', [])
        logger.info(f"Existing aliases for agent '{agent_name}': {existing_aliases}")

        target_alias = next((a for a in existing_aliases if a['agentAliasName'] == agent_alias), None)

        if target_alias:
            logger.info(f"Found existing alias: {target_alias['agentAliasName']} with ID: {target_alias['agentAliasId']}")

            if target_alias['agentAliasStatus'] != "PREPARED":
                logger.info(f"Alias '{agent_alias}' is not prepared. Waiting for it to be prepared...")
                llm_utils.wait_for_agent_alias_status(
                    agent_id=agent_id,
                    agent_alias_id=target_alias['agentAliasId'],
                    target_status="PREPARED"
                )
            agent_alias_id = target_alias['agentAliasId']
        else:
            # Create the alias if it doesn't exist
            logger.info("Creating a new agent alias...")
            create_agent_alias_response = llm_utils.bedrock_agent.create_agent_alias(
                agentId=agent_id,
                agentAliasName=agent_alias
            )
            agent_alias_id = create_agent_alias_response['agentAlias']['agentAliasId']
            logger.info(f"Agent alias created with ID: {agent_alias_id}")

            # Wait for the agent alias to be prepared
            llm_utils.wait_for_agent_alias_status(
                agent_id=agent_id,
                agent_alias_id=agent_alias_id,
                target_status="PREPARED"
            )

        logger.info("Bootstrap completed successfully. Agent and alias are ready.")
    except Exception as e:
        logger.error(f"Failed during bootstrap: {str(e)}")
        raise RuntimeError("Bootstrap failed.")


try:
    logger.info("Initializing the Lambda container with bootstrap...")
    bootstrap()
except Exception as e:
    logger.error(f"Failed to bootstrap during container initialization: {str(e)}")
    raise RuntimeError("Failed to bootstrap the Lambda container.")


def handler(event, context):
    """
    AWS Lambda handler to invoke the Bedrock agent with user input.

    Parameters:
    - event (dict): The incoming event payload.
    - context (object): The AWS Lambda context object.

    Returns:
    - dict: A response object with the agent's output.
    """
    global agent_id, agent_alias_id
    logger.info(f"Received event: {event}")

    try:
        # Extract session ID and input text from the event
        session_id = event.get('sessionId', str(uuid.uuid4()))
        input_text = event.get('inputText', '')

        if not input_text:
            raise ValueError("Input text is missing.")

        # Check if the agent and alias are prepared
        if not llm_utils.is_ready(agentId=agent_id, agentAliasId=agent_alias_id):
            return {
                "statusCode": 503,
                "body": json.dumps({"error": "Agent or alias not ready to serve requests."})
            }

        # Invoke the Bedrock agent
        logger.info("Invoking Bedrock agent...")
        final_response = llm_utils.invoke_agent(
            agentId=agent_id,
            agentAliasId=agent_alias_id,
            inputText=input_text,
            sessionId=session_id,
            enableTrace=agentTrace
        )

        # Extract and return the agent's response
        if not final_response:
            raise ValueError("Agent response is empty or malformed.")

        # Return the response to the client
        return {
            "statusCode": 200,
            "body": json.dumps({
                "agentResponse": final_response,
                "sessionId": session_id
            })
        }
    except Exception as e:
        logger.error(f"Failed to invoke agent: {str(e)}")
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }



