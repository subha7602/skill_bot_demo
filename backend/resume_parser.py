import os
import re
import json
import fitz  
import uuid
from typing import List, Dict, Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError
from langchain.vectorstores.faiss import FAISS
from langchain.schema import Document
from langchain_aws import BedrockEmbeddings
from langchain_community.chat_models.bedrock import BedrockChat
from langchain_core.messages import HumanMessage, SystemMessage
from shared_models import Profile

class ResumeParserLangchain:
    def __init__(self, resume_directory: str):
        self.resume_directory = resume_directory
        
        load_dotenv()

        if not (os.getenv('AWS_ACCESS_KEY_ID') and os.getenv('AWS_SECRET_ACCESS_KEY') and os.getenv('AWS_REGION')):
            raise ValueError("AWS credentials not found in environment variables")
            
        # Initialize Bedrock LLM
        self.llm = BedrockChat(
            model_id="anthropic.claude-3-haiku-20240307-v1:0", 
            region_name=os.getenv('AWS_REGION'),
            model_kwargs={
                "temperature": 0,
                "max_tokens": 5000
            }
        )
        
        self.system_prompt = """
        Extract resume information and return it in the following JSON format:
        {
            "name": "Full Name",
            "job_title": "Current Job Title",
            "email": "email@example.com",
            "linkedin": "LinkedIn URL",
            "location": "City, State",
            "executive_summary": "Brief professional summary",
            "skills": ["Skill 1", "Skill 2", ...],
            "projects": [
                {
                    "project_role": "Project role",
                    "project_name": "Project Name",
                    "start_date": "YYYY-MM-DD",
                    "end_date": "YYYY-MM-DD",  // or null if current position
                    "responsibilities": ["Responsibility 1", "Responsibility 2", ...]
                }
            ],
            "certifications": [
                {
                    "certification_name": "Certification Name",
                    "certificate_provider": "Certification Provider",
                    "level": "Level (e.g., Associate, Professional)"
                }
            ]
        }

        Guidelines:
        1. Dates should be in YYYY-MM-DD format. If only month and year are available, use the first day of the month.
        2. If a field is not found in the resume, use null or empty list [] as appropriate.
        3. Return only the JSON object, no markdown or code blocks.
        4. Ensure all text fields use proper capitalization and punctuation.
        5. Include all relevant skills mentioned in the resume.
        6. Convert job responsibilities into clear, concise statements.
        7. When extracting names, provide the full name with proper spacing. Preserve standard naming conventions.
        8. Project Details:
           - project_role: This is the specific role the individual held within the project (e.g., DevOps Engineer, Cloud Migration Specialist).
           - project_name: This refers to the specific name of the project. If the project name is not given, use a descriptive name that explains the type of project (e.g., "AWS Migration").
           - responsibilities: List clear, concise tasks or responsibilities for each project. These should be specific and relate to the individual's contributions within the project.
        9. For certifications, extract:
           - certification_name: The name of the certification (e.g., "Terraform Associate")
           - certificate_provider: The organization that provides the certification (e.g., "HashiCorp")
           - level: The level of certification if mentioned (e.g., "Associate", "Professional")
        10. When parsing certifications like "HashiCorp Certified: Terraform Associate", separate the provider and certification name appropriately.
        11. Current Projects: If the individual is currently working on a project, set the `end_date` as `null` to indicate it's ongoing. This helps differentiate between past and present projects.
        """

    def extract_json_from_response(self, response_text: str) -> str:
        """Extract JSON from response that might be wrapped in markdown code blocks."""
        
        json_match = re.search(r'```(?:json)?\n?(.*?)\n?```', response_text, re.DOTALL)
        if json_match:
            return json_match.group(1).strip()
        return response_text.strip()

    def clean_text(self, text: str) -> str:
        """Clean extracted text while preserving proper spacing of words."""
        # Replace non-ASCII characters
        text = re.sub(r"[^\x00-\x7F]+", " ", text)
        
        # Normalize whitespace - first collapse multiple spaces into one
        text = re.sub(r"\s+", " ", text)
        
        return text.strip()
    
    def load_single_resume(self, file_path: str) -> str:
        """Load and extract text from a PDF resume using PyMuPDF (fitz)."""
        try:
            with fitz.open(file_path) as doc:
                full_text = " ".join(page.get_text("text") for page in doc)
            return self.clean_text(full_text)
        except Exception as e:
            print(f"Error loading {file_path}: {str(e)}")
            return ""

    def load_all_resumes(self) -> List[str]:
        """Load all resumes from the directory."""
        raw_texts = []
        for file_name in os.listdir(self.resume_directory):
            if file_name.endswith(".pdf"):
                file_path = os.path.join(self.resume_directory, file_name)
                text = self.load_single_resume(file_path)
                if text:
                    raw_texts.append(text)
        return raw_texts

    def process_text(self, text: str) -> Optional[dict]:
        """Process a single text through the LLM."""
        try:
            messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=text)
            ]
            
            response = self.llm.invoke(messages)
            
            # Extract and clean JSON from the response
            json_str = self.extract_json_from_response(response.content)
            
            # Parse JSON string into dict
            json_data = json.loads(json_str)
                       
            # Validate using Pydantic
            profile_data = Profile.model_validate(json_data)
            return profile_data.model_dump(exclude_none=True)
            
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {str(e)}")
            print(f"Raw response: {response.content}")
            return None
        except Exception as e:
            print(f"Error processing text: {str(e)}")
            return None

    def process_all_resumes(self) -> List[dict]:
        """Process all resumes in the directory."""
        raw_texts = self.load_all_resumes()
        results = []
        for text in raw_texts:
            result = self.process_text(text)
            print(result)
            if result:
                print(f"Processed resume: {result.get('name', 'Unnamed')}")
                results.append(result)
        
        return results

   