import os
import re
import json
import uuid
import fitz
from typing import List, Dict, Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError
from langchain_openai import OpenAIEmbeddings
from langchain_aws import BedrockEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models.bedrock import BedrockChat
from langchain.schema import Document
from langchain_core.messages import HumanMessage, SystemMessage
from resume_parser import ResumeParserLangchain
from shared_models import Profile

class ResumeVectorStore:
    def __init__(self):
        """Initialize the ResumeVectorStore with Bedrock embeddings."""
        load_dotenv()
        
        # Check for AWS credentials
        if not (os.getenv('AWS_ACCESS_KEY_ID') and os.getenv('AWS_SECRET_ACCESS_KEY') and os.getenv('AWS_REGION')):
            raise ValueError("AWS credentials not found in environment variables")
            
        # Use Bedrock embeddings instead of OpenAI
        self.embeddings = BedrockEmbeddings(
            model_id="amazon.titan-embed-text-v1",  # Use appropriate Bedrock embedding model
            region_name=os.getenv('AWS_REGION')
        )

    def clean_name(self, name: str) -> str:
        """Cleans and normalizes the extracted name."""
        if not name:
            return "Unknown"
        cleaned_name = re.sub(r"[^a-zA-Z\s]", "", name).strip()
        return " ".join(cleaned_name.split())  # Normalize spacing

    def create_documents_from_json(self, profile_data: Dict) -> List[Document]:
        """Convert JSON profile data into Document objects for embedding."""
        documents = []
        metadata_id = str(uuid.uuid4())
        profile_name = self.clean_name(profile_data.get('name', 'Unknown'))

        # Add certifications to the main content
        certifications_text = ""
        if profile_data.get('certifications'):
            certifications_text = "Certifications:\n"
            for cert in profile_data.get('certifications', []):
                cert_text = f"{cert.get('certificate_provider', '')} {cert.get('certification_name', '')}"
                if cert.get('level'):
                    cert_text += f" ({cert.get('level', '')})"
                certifications_text += f"• {cert_text}\n"

        main_content = f"""
        Name: {profile_name}
        Job Title: {profile_data.get('job_title', 'Unknown')}
        Location: {profile_data.get('location', 'Unknown')}
        Executive Summary: {profile_data.get('executive_summary', 'No Summary Provided')}
        Skills: {', '.join(profile_data.get('skills', []))}
        {certifications_text}
        """

        documents.append(Document(
            page_content=main_content,
            metadata={
                "id": metadata_id,  
                "type": "profile",
                "name": profile_name,
                "skills": ', '.join(profile_data.get('skills', [])),
                "doc_type": "main_profile"
            }
        ))
        
        # Create separate documents for each certification
        for cert in profile_data.get('certifications', []):
            cert_content = f"""
            Certification: {cert.get('certification_name', 'Unknown')}
            Provider: {cert.get('certificate_provider', 'Unknown')}
            Level: {cert.get('level', 'N/A')}
            """

            documents.append(Document(
                page_content=cert_content,
                metadata={
                    "id": metadata_id,  
                    "type": "certification",
                    "name": profile_name,
                    "certification": cert.get('certification_name', 'Unknown'),
                    "provider": cert.get('certificate_provider', 'Unknown'),
                    "doc_type": "certification"
                }
            ))
        
        # Existing code for projects
        for project in profile_data.get('projects', []):
            project_content = f"""
            Project Role: {project.get('project_role', 'Unknown')}
            Project Name: {project.get('project_name', 'Unknown')}
            Period: {project.get('start_date', 'N/A')} to {project.get('end_date', 'Present')}
            Responsibilities:
            {' '.join(project.get('responsibilities', []))}
            """

            documents.append(Document(
                page_content=project_content,
                metadata={
                    "id": metadata_id,
                    "type": "project",
                    "name": profile_name,
                    "project_name": project.get('project_name', 'Unknown'),
                    "project_role": project.get('project_role', 'Unknown'),
                    "doc_type": "project"
                }
            ))

        
        return documents

    def create_vector_store(self, json_profiles: List[Dict], save_path: str = "faiss_index") -> Optional[FAISS]:
        """Create FAISS vector store from JSON profile data."""
        try:
            all_documents = []
            
            for profile_data in json_profiles:
                profile = Profile.model_validate(profile_data)
                documents = self.create_documents_from_json(profile.model_dump(exclude_none=True))
                all_documents.extend(documents)
            
            if not all_documents:
                raise ValueError("No valid profile data to process")
            
            vector_store = FAISS.from_documents(all_documents, self.embeddings)
            vector_store.save_local(save_path)
            print(f"\nVector store successfully created and saved at '{save_path}'")
            
            return vector_store
            
        except ValidationError as ve:
            print(f"Validation Error: {ve}")
            return None
        except Exception as e:
            print(f"Error creating vector store: {str(e)}")
            return None
    
    def load_vector_store(self, load_path: str = "faiss_index") -> Optional[FAISS]:
        """Load an existing FAISS vector store."""
        try:
            vector_store = FAISS.load_local(load_path, self.embeddings)
            print(f"Vector store loaded from '{load_path}'")
            return vector_store
        except Exception as e:
            print(f"Error loading vector store: {str(e)}")
            return None

def create_faiss_index(resume_dir: str = "resumes/", index_path: str = "faiss_index") -> None:
    """Create FAISS index from resume PDFs in the specified directory."""
    load_dotenv()
    
    # Check for AWS credentials
    if not (os.getenv('AWS_ACCESS_KEY_ID') and os.getenv('AWS_SECRET_ACCESS_KEY') and os.getenv('AWS_REGION')):
        raise ValueError("AWS credentials not found in environment variables")
    
    try:
        print("Parsing resumes...")
        parser = ResumeParserLangchain(resume_directory=resume_dir)
        json_profiles = parser.process_all_resumes()
        
        if not json_profiles:
            raise ValueError(f"No valid profiles found in {resume_dir}")
        
        print(f"Successfully parsed {len(json_profiles)} resumes")
        
        print("\nCreating vector store...")
        vector_store = ResumeVectorStore()
        result = vector_store.create_vector_store(
            json_profiles=json_profiles,
            save_path=index_path
        )
        
        if result:
            print(f"\nSuccess! FAISS index created and saved to {index_path}")
        else:
            raise ValueError("Failed to create vector store")
            
    except Exception as e:
        print(f"Error creating FAISS index: {str(e)}")
        raise

if __name__ == "__main__":
    if not os.path.exists("resumes"):
        os.makedirs("resumes")
        print("Created 'resumes' directory")
        print("Please add PDF resumes to the 'resumes' directory before running this script")
    else:
        create_faiss_index()