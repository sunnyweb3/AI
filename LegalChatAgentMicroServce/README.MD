# AWS Bedrock AI Agent Framework - README

## Overview

This project demonstrates how to deploy scalable, configurable, and efficient AI agents using **AWS Bedrock AI Agentic Framework** and **AWS Lambda with Docker**. The framework is flexible and adaptable, enabling users to build specific AI agents, such as a Legal Assistant, Customer Support Agent, or Knowledge Retrieval Bot.

## Features

- **Leverages AWS Bedrock Models**: Utilize foundational models like `Nova Lite` and `Titan Text` for natural language understanding.
- **Agent Customization**: Modify `llmconfig.json` and `system_instructions.txt` to quickly adapt the agent to various use cases.
- **Stateful Conversations**: Use session IDs to maintain context across multi-turn conversations.
- **Dynamic Model Selection**: Switch foundational models to handle tasks of varying complexity.

## File Structure

- **`app.py`**: Main application logic for managing the agent lifecycle and serving requests.
- **`llmutil.py`**: Helper utility for interacting with AWS Bedrock APIs.
- **`llmconfig.json`**: Configuration file for agent parameters like model, instructions, and roles.
- **`system_instructions.txt`**: Defines agent behavior and scope.
- **`requirements.txt`**: Lists Python dependencies.
- **`dockerfile`**: Instructions to containerize the application for AWS Lambda.

## Setup and Deployment

### Prerequisites

1. AWS account with permissions for AWS Bedrock and Lambda.
2. Docker installed locally for building the container image.
3. Python 3.9 or compatible runtime for local development.

### Configuration

1. **Agent Behavior**:
   - Modify `system_instructions.txt` to set the agent's scope (e.g., legal assistant, customer support).
2. **Agent Parameters**:
   - Update `llmconfig.json` to define:
     - `agentName`: Unique identifier for the agent.
     - `foundationModel`: AWS Bedrock model to use (e.g., `amazon.nova-lite-v1:0`).
     - `instruction`: Tailored prompts for the agent.
3. **Dependencies**:
   - Add additional Python libraries to `requirements.txt` if needed.

### Deployment Steps

1. **Build Docker Image**:
   ```bash
   docker build -t bedrock-agent .
   ```

2. **Deploy to AWS Lambda**:
   - Use the AWS CLI or Management Console to upload the container image.
   - Set the Lambda handler to `app.handler`.

3. **Test the Agent**:
   - Use Postman or the AWS Lambda test interface to verify responses.

## Usage

### Example API Call

- **Request**:
  ```json
  {
    "sessionId": "12345",
    "inputText": "What are the key elements of a contract?"
  }
  ```
- **Response**:
  ```json
  {
    "agentResponse": "A contract typically includes offer, acceptance, consideration, and mutual agreement.",
    "sessionId": "12345"
  }
  ```

### Enhancements

- Add multimedia processing by integrating OCR and audio transcription services.
- Extend functionality by connecting to external APIs or databases.

## Resources

- **Amazon Bedrock Documentation**: [Explore Bedrock](https://aws.amazon.com/bedrock)
- **Postman Collection**: [Test Your Agent](#)
- **GitHub Repository**: [Access the Source Code](#)
