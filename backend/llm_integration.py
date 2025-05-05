from datetime import datetime
import os
import re
import json
import time
import random
import traceback
from typing import Dict, List, Any, Optional, Set, Tuple
from functools import lru_cache
from collections import defaultdict
import difflib

import numpy as np
from tenacity import retry, wait_exponential, stop_after_attempt
from dotenv import load_dotenv
import boto3

# LangChain imports
from langchain.schema import Document
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate 
from langchain_aws import BedrockEmbeddings
from langchain_aws import ChatBedrock
from langchain.schema import SystemMessage, HumanMessage
class StringSimilarity:
    """
    Enhanced string similarity calculations with optimized implementations.
    """
    @staticmethod
    def ratio(str1, str2):
        """Calculate a similarity ratio between two strings."""
        if not str1 and not str2:  # Both empty
            return 100
        if not str1 or not str2:  # One empty
            return 0
            
        # Use quick length check before expensive comparison
        len1, len2 = len(str1), len(str2)
        if abs(len1 - len2) / max(len1, len2) > 0.5:
            return 0  # Significant length difference, unlikely to be similar
            
        return int(difflib.SequenceMatcher(None, str1, str2).ratio() * 100)
        
    @staticmethod
    def partial_ratio(str1, str2):
        """Calculate partial string similarity by finding best matching substring."""
        # Handle empty strings
        if not str1 and not str2:  # Both empty
            return 100
        if not str1 or not str2:  # One empty
            return 0
            
        # Find shorter and longer strings
        if len(str1) <= len(str2):
            shorter, longer = str1, str2
        else:
            shorter, longer = str2, str1
            
        # Early return for efficiency 
        if not shorter:
            return 100 if not longer else 0
            
        # Optimization: If shorter is significantly smaller than longer,
        # limit the number of windows to check based on potential match quality
        len_shorter, len_longer = len(shorter), len(longer)
        
        # If shorter string is very small compared to longer, only check a reasonable number of windows
        # to avoid unnecessary computation
        if len_shorter < len_longer * 0.2 and len_longer > 100:
            # Check windows with some potential shared content
            matches = []
            words_shorter = set(shorter.lower().split())
            
            # Sliding window with word-based pre-filtering
            for i in range(0, len_longer - len_shorter + 1, max(1, len_shorter // 2)):
                window = longer[i:i+len_shorter]
                
                # Quick pre-check: if window shares any words with shorter string
                if any(word in window.lower() for word in words_shorter):
                    matches.append(StringSimilarity.ratio(shorter, window))
                    
            # If no potential matches found with pre-filtering, check some random windows
            if not matches and len_longer > len_shorter:
                # Check a few strategic positions (start, middle, end)
                positions = [0, len_longer // 2 - len_shorter // 2, len_longer - len_shorter]
                for pos in positions:
                    matches.append(StringSimilarity.ratio(shorter, longer[pos:pos+len_shorter]))
                
            return max(matches) if matches else 0
        else:
            # Standard sliding window for similarly sized strings
            matches = []
            step_size = 1
            
            # For very long strings, increase the step size to improve performance
            if len_longer > 1000:
                step_size = max(1, len_shorter // 10)
                
            for i in range(0, len_longer - len_shorter + 1, step_size):
                window = longer[i:i+len_shorter]
                matches.append(StringSimilarity.ratio(shorter, window))
                
            return max(matches) if matches else 0
        
    @staticmethod
    def token_sort_ratio(str1, str2):
        """Sort words and compare."""
        # Handle empty strings
        if not str1 and not str2:  # Both empty
            return 100
        if not str1 or not str2:  # One empty
            return 0
            
        # Optimization: Quick check for exact matches
        if str1 == str2:
            return 100
        
        # Sort words and compare
        sorted1 = ' '.join(sorted(str1.split()))
        sorted2 = ' '.join(sorted(str2.split()))
        return StringSimilarity.ratio(sorted1, sorted2)
        
    @staticmethod
    def token_set_ratio(str1, str2):
        """Compare sets of words in the strings with optimized implementation."""
        # Handle empty strings
        if not str1 and not str2:  # Both empty
            return 100
        if not str1 or not str2:  # One empty
            return 0
            
        # Optimization: Quick check for exact matches
        if str1 == str2:
            return 100
            
        # Split into words and create sets
        set1 = set(str1.split())
        set2 = set(str2.split())
        
        # Early optimization: if sets are identical
        if set1 == set2:
            return 100
            
        # Common words
        intersection = set1.intersection(set2)
        
        # Early return if no common words
        if not intersection:
            return 0
        
        # Words unique to each string
        diff1 = set1.difference(set2)
        diff2 = set2.difference(set1)
        
        # Construct sorted strings
        sorted_sect = ' '.join(sorted(intersection))
        sorted_diff1 = ' '.join(sorted(diff1))
        sorted_diff2 = ' '.join(sorted(diff2))
        
        # Different combinations of the parts to find highest similarity
        combined_1 = f"{sorted_sect} {sorted_diff1}" if sorted_diff1 else sorted_sect
        combined_2 = f"{sorted_sect} {sorted_diff2}" if sorted_diff2 else sorted_sect
        
        # Compute ratios
        ratio1 = StringSimilarity.ratio(sorted_sect, combined_1)
        ratio2 = StringSimilarity.ratio(sorted_sect, combined_2)
        ratio3 = StringSimilarity.ratio(combined_1, combined_2)
        
        # Return the maximum ratio
        return max(ratio1, ratio2, ratio3)

    @staticmethod
    def weighted_similarity(str1, str2):
        """
        Calculate a weighted similarity score combining multiple metrics.
        This provides a more balanced similarity assessment.
        
        Args:
            str1: First string to compare
            str2: Second string to compare
            
        Returns:
            int: Weighted similarity score (0-100)
        """
        # Handle empty strings
        if not str1 and not str2:  # Both empty
            return 100
        if not str1 or not str2:  # One empty
            return 0
        
        # Calculate each metric
        ratio = StringSimilarity.ratio(str1, str2)
        partial_ratio = StringSimilarity.partial_ratio(str1, str2)
        token_sort = StringSimilarity.token_sort_ratio(str1, str2)
        token_set = StringSimilarity.token_set_ratio(str1, str2)
        
        # Weighted combination for a balanced score
        weighted_score = int((ratio * 0.25) + 
                            (partial_ratio * 0.35) + 
                            (token_sort * 0.2) + 
                            (token_set * 0.2))
                            
        return weighted_score

class StringProcess:
    """
    Improved string processing utilities with performance optimizations
    """
    @staticmethod
    def extract(query, choices, limit=5, min_score=40):
        """
        Enhanced extract function that mimics fuzz.process.extract with better performance.
        
        Args:
            query: String to match against choices
            choices: List of strings to match against
            limit: Maximum number of results to return
            min_score: Minimum similarity score to include in results
            
        Returns:
            List[Tuple[str, int]]: List of (choice, score) tuples
        """
        if not query or not choices:
            return []
            
        results = []
        query_lower = query.lower()
        
        # Optimization: Pre-filter candidates for long lists
        if len(choices) > 100:
            # Quick pre-filtering using substring containment
            initial_candidates = []
            
            # Get query words for filtering
            query_words = set(query_lower.split())
            
            for choice in choices:
                choice_lower = choice.lower()
                
                # Check if any query word exists in the choice
                if any(word in choice_lower for word in query_words if len(word) > 2):
                    initial_candidates.append(choice)
                # Also check if choice contains part of query
                elif any(word in query_lower for word in choice_lower.split() if len(word) > 2):
                    initial_candidates.append(choice)
                    
            # If we found reasonable candidates, use them instead of all choices
            if len(initial_candidates) >= limit and len(initial_candidates) < len(choices) // 2:
                choices = initial_candidates
            # If we have too many candidates, keep some original choices
            elif len(initial_candidates) > len(choices) // 2:
                # Keep a mix of filtered candidates and original choices
                choices = initial_candidates[:len(choices) // 2]
        
        # For each potential match, calculate comprehensive similarity
        for choice in choices:
            # Use the new weighted similarity for more accurate results
            score = StringSimilarity.weighted_similarity(query, choice)
            
            # Only consider scores above minimum threshold
            if score >= min_score:
                results.append((choice, score))
            
        # Sort by score and limit results
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]


class ContextualProjectRetrieval:
    """
    A comprehensive system for contextual retrieval of project information.
    """
    def __init__(self, vector_store_path="faiss_index", aws_region=None):
        """
        Initialize the retrieval system.
        
        Args:
            vector_store_path: Path to the FAISS index
            aws_region: AWS region for Bedrock services
        """
        # Load environment variables
        self.load_env_from_path("/etc/secrets/env")
    
        
        # Set AWS region (use environment or passed value)
        self.aws_region = aws_region or os.getenv('AWS_REGION', 'us-east-1')
        
        # Initialize Bedrock client
        self.bedrock_client = boto3.client(
            "bedrock-runtime", 
            region_name=self.aws_region
        )
        
        # Initialize embeddings
        self.embeddings = BedrockEmbeddings(
            model_id="amazon.titan-embed-text-v1",
            region_name=self.aws_region
        )
        
        # Load vector store
        try:
            if os.path.exists(vector_store_path):
                self.vector_store = FAISS.load_local(
                    vector_store_path, 
                    self.embeddings, 
                    allow_dangerous_deserialization=True
                )
                print(f"Loaded FAISS index from {vector_store_path}")
            else:
                raise FileNotFoundError(f"FAISS index not found at {vector_store_path}")
        except Exception as e:
            print(f"Error loading FAISS index: {e}")
            raise
        
        # Cache for term context vectors (compute once per session)
        self.term_context_vectors_cache = None
        self.StringSimilarity = StringSimilarity
    
    # Initialize name matching functionality
        self.initialize_name_matching()
    
    def load_env_from_path(self, env_file_path):
        """
        Load environment variables from specified file path.
        
        Args:
            env_file_path: Path to the environment file
        """
        try:
            if not os.path.exists(env_file_path):
                print(f"Warning: Environment file not found at {env_file_path}")
                return
                
            with open(env_file_path, 'r') as file:
                for line in file:
                    line = line.strip()
                    # Skip empty lines and comments
                    if not line or line.startswith('#'):
                        continue
                        
                    # Parse key-value pairs
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip().strip('"\'')
                    
            print(f"Loaded environment variables from {env_file_path}")
        except Exception as e:
            print(f"Error loading environment variables: {e}")

    @retry(wait=wait_exponential(min=1, max=60), stop=stop_after_attempt(5))
    def get_embedding_with_retry(self, text):
        """Get embeddings with retry logic for rate limits"""
        try:
            return self.embeddings.embed_query(text)
        except Exception as e:
            print(f"Error getting embeddings: {e}")
            time.sleep(random.uniform(1, 3))
            raise
    
    @lru_cache(maxsize=1000)
    def cached_embedding(self, text):
        """Cache embeddings to reduce API calls"""
        return self.get_embedding_with_retry(text)
    
    @retry(wait=wait_exponential(min=1, max=30), stop=stop_after_attempt(3))
    def invoke_bedrock_with_retry(self, chain, params):
        """Invoke Bedrock with retry logic"""
        try:
            return chain.invoke(params)
        except Exception as e:
            print(f"Error invoking Bedrock: {e}. Retrying...")
            time.sleep(random.uniform(1, 2))
            raise
    
    def cosine_similarity_calc(self, vector_a, vector_b):
            """Calculate cosine similarity between two vectors with improved robustness."""
            try:
                # Convert to numpy arrays if they aren't already
                a = np.array(vector_a, dtype=np.float32)
                b = np.array(vector_b, dtype=np.float32)
                
                # Check for NaN or infinity values
                if np.isnan(a).any() or np.isinf(a).any() or np.isnan(b).any() or np.isinf(b).any():
                    print("Warning: NaN or infinity values in vectors. Replacing with zeros.")
                    a = np.nan_to_num(a)
                    b = np.nan_to_num(b)
                
                # Ensure both are 2D arrays
                if a.ndim == 1:
                    a = a.reshape(1, -1)
                if b.ndim == 1:
                    b = b.reshape(1, -1)
                
                # Normalize the vectors
                a_norm = np.linalg.norm(a, axis=1, keepdims=True)
                b_norm = np.linalg.norm(b, axis=1, keepdims=True)
                
                # Avoid division by zero
                a_norm = np.where(a_norm == 0, 1e-10, a_norm)
                b_norm = np.where(b_norm == 0, 1e-10, b_norm)
                
                a_normalized = a / a_norm
                b_normalized = b / b_norm
                
                # Calculate similarity
                similarity = np.dot(a_normalized, b_normalized.T)
                
                # If the input was single vectors, return a float
                if similarity.shape == (1, 1):
                    return float(similarity[0, 0])
                else:
                    return similarity
            except Exception as e:
                print(f"Error in cosine similarity calculation: {e}")
                # Return -1 to indicate error (instead of crashing)
                return -1.0 
            
    def get_all_resumes_from_store(self):
        """Retrieve all resumes from the FAISS vector store."""
        try:
            doc_ids = list(self.vector_store.docstore._dict.keys())
            documents = [self.vector_store.docstore._dict[doc_id] for doc_id in doc_ids]
            context = "\n\n---RESUME SEPARATOR---\n\n".join(doc.page_content for doc in documents)
            return context
        except Exception as e:
            print(f"Error retrieving all resumes: {e}")
            return ""
    
    def chunk_context(self, context, max_chunk_size=8000):
        """Split resumes into chunks with improved logic."""
        if not context:
            return []
            
        # Split by resume separator
        resumes = context.split("---RESUME SEPARATOR---")
        chunks = []
        current_chunk = []
        current_length = 0
        
        for resume in resumes:
            if not resume.strip():  # Skip empty resumes
                continue
                
            resume_length = len(resume)
            
            # If this resume alone exceeds max size, split it further
            if resume_length > max_chunk_size:
                # If we have accumulated content, add it as a chunk
                if current_chunk:
                    chunks.append("\n\n---RESUME SEPARATOR---\n\n".join(current_chunk))
                    current_chunk = []
                    current_length = 0
                
                # Split the long resume into smaller chunks
                resume_parts = []
                remaining = resume
                while len(remaining) > max_chunk_size:
                    # Find a good split point (paragraph or sentence break)
                    split_point = max_chunk_size
                    
                    # Try to split at a paragraph
                    paragraph_split = remaining[:split_point].rfind("\n\n")
                    if paragraph_split > max_chunk_size * 0.7:  # At least 70% of max size
                        split_point = paragraph_split
                    else:
                        # Try to split at a sentence
                        sentence_end = re.finditer(r'[.!?]\s', remaining[:split_point])
                        last_sentence = None
                        for match in sentence_end:
                            last_sentence = match.end()
                            
                        if last_sentence and last_sentence > max_chunk_size * 0.7:
                            split_point = last_sentence
                    
                    # Add this part and continue with remainder
                    resume_parts.append(remaining[:split_point])
                    remaining = remaining[split_point:]
                
                # Add the last part
                if remaining:
                    resume_parts.append(remaining)
                
                # Add each part as its own chunk
                for part in resume_parts:
                    chunks.append(part)
            
            # Normal case: add to current chunk if fits
            elif current_length + resume_length > max_chunk_size and current_chunk:
                chunks.append("\n\n---RESUME SEPARATOR---\n\n".join(current_chunk))
                current_chunk = [resume]
                current_length = resume_length
            else:
                current_chunk.append(resume)
                current_length += resume_length
        
        # Add the final chunk if not empty
        if current_chunk:
            chunks.append("\n\n---RESUME SEPARATOR---\n\n".join(current_chunk))
        
        return chunks
    
        return chunks
    
    def retrieve_resumes(self, query, top_k=25, search_type="all"):
            """Retrieve top-k resumes based on FAISS search and keyword filtering with improved logic."""
            if not query or not isinstance(query, str):
                return []
                
            enhanced_query = f"Find resumes that match the following criteria: {query}"
            
            # Use cached embeddings with retry logic
            try:
                # Step 1: FAISS Semantic Search
                results = self.vector_store.similarity_search(enhanced_query, k=top_k)
                
                if not results:
                    return []
                    
                # Step 2: Keyword-based filtering (ensure strict matching)
                search_terms = set(query.lower().split())
                
                # Remove very common words from search terms
                stopwords = {'and', 'or', 'the', 'a', 'an', 'is', 'are', 'in', 'on', 'at', 'for', 'with', 'to', 'by'}
                search_terms = {term for term in search_terms if term not in stopwords and len(term) > 1}
                
                # Handle different search types
                if search_type == "skills":
                    matching_resumes = [
                        result for result in results 
                        if any(term in result.metadata.get('skills', '').lower() for term in search_terms)
                    ]
                elif search_type == "projects":
                    matching_resumes = [
                        result for result in results 
                        if any(term in result.metadata.get('projects', '').lower() for term in search_terms)
                    ]

                else:  # "all"
                    matching_resumes = [
                        result for result in results 
                        if (any(term in result.metadata.get('skills', '').lower() for term in search_terms) or
                            any(term in result.metadata.get('projects', '').lower() for term in search_terms) 
                           )
                    ]
                
                # If no exact keyword matches, return top FAISS results
                if not matching_resumes:
                    matching_resumes = results
                
                # Extract employee names and ensure uniqueness
                employee_names = []
                seen_names = set()
                
                for resume in matching_resumes:
                    name = resume.metadata.get('name')
                    if name and name not in seen_names:
                        employee_names.append(name)
                        seen_names.add(name)
                
                return employee_names
            except Exception as e:
                print(f"Error retrieving resumes: {e}")
                traceback.print_exc()
                return []
        
    def create_project_document_store(self):
        """
        Create a structured document store for projects from the vector store.
        Improved implementation with error handling for edge cases.
        
        Returns:
            dict: A structured project document store
        """
        project_store = {
            "by_person": {},  # Projects indexed by person
            "by_project": {},  # People indexed by project
            "all_projects": [],  # List of all projects with full details
            "metadata": {
                "people_count": 0,
                "project_count": 0,
               
            }
        }
        
        try:
            # Get all documents from vector store
            doc_ids = list(self.vector_store.docstore._dict.keys())
            documents = [self.vector_store.docstore._dict[doc_id] for doc_id in doc_ids]
            
            # Process each document
            for doc in documents:
                # Only process project documents
                if doc.metadata.get('doc_type') == 'project':
                    person_name = doc.metadata.get('name', 'Unknown')
                    project_name = doc.metadata.get('project_name', 'Unknown Project')
                    project_role = doc.metadata.get('project_role', 'Unknown Role')
                 
                    
                    # Skip if missing required fields
                    if not person_name or person_name == 'Unknown' or not project_name or project_name == 'Unknown Project':
                        continue
                    
                    # Extract additional metadata and content
                    project_details = {
                        'name': person_name,
                        'project_name': project_name,
                        'project_role': project_role,
                        'content': doc.page_content,
                        'metadata': doc.metadata
                    }
                    
                    # Index by person
                    if person_name not in project_store['by_person']:
                        project_store['by_person'][person_name] = []
                        project_store['metadata']['people_count'] += 1
                    
                    # Avoid duplicates for the same person-project combination
                    if not any(p['project_name'] == project_name for p in project_store['by_person'][person_name]):
                        project_store['by_person'][person_name].append(project_details)
                    
                    # Index by project
                    if project_name not in project_store['by_project']:
                        project_store['by_project'][project_name] = []
                        project_store['metadata']['project_count'] += 1
                    
                    # Avoid duplicates for the same person in a project
                    if not any(p['name'] == person_name for p in project_store['by_project'][project_name]):
                        project_store['by_project'][project_name].append(project_details)
                    
                    # Add to all projects
                    project_store['all_projects'].append(project_details)
                    

                    
            print(f"Created project document store with {len(project_store['all_projects'])} projects")
            print(f"Indexed {project_store['metadata']['people_count']} people with projects")
            print(f"Indexed {project_store['metadata']['project_count']} unique projects")
         
            

            
            return project_store
        
        except Exception as e:
            print(f"Error creating project document store: {e}")
            traceback.print_exc()
            return {"by_person": {}, "by_project": {}, "all_projects": [], 
                "metadata": {"people_count": 0, "project_count": 0}}
    
    def extract_key_terms(self, query: str) -> List[str]:
        """
        Enhanced method to extract key technical terms and entities from a query.
        
        Args:
            query: The user query
            
        Returns:
            List[str]: List of extracted key terms
        """
        if not query:
            return []
            
        query_lower = query.lower()
        
        # Extract terms using various methods
        terms = set()
        
        # Method 1: Technical terms detection using regex patterns
        tech_patterns = [
            # Cloud platforms and services
            r'\b(aws|amazon|azure|microsoft|gcp|google cloud|lambda|s3|ec2|dynamodb|cosmos|cloud)\b',
            # Infrastructure and DevOps
            r'\b(kubernetes|k8s|docker|terraform|jenkins|gitlab|github|ci/cd|cicd|pipeline|deployment)\b',
            # Development technologies
            r'\b(python|java|javascript|node\.js|react|angular|vue|express|spring|django|flask)\b',
            # Concepts and roles
            r'\b(devops|sre|cloud native|infrastructure|migration|architect|security|monitoring)\b',
            # AI and ML terms
            r'\b(machine learning|ml|ai|artificial intelligence|deep learning|nlp|neural network)\b',
            # Database technologies
            r'\b(sql|mysql|postgresql|mongodb|nosql|database|db|oracle|sqlite)\b',
            # Mobile development
            r'\b(android|ios|swift|kotlin|flutter|react native|mobile)\b',
            # Web technologies
            r'\b(html|css|javascript|frontend|backend|fullstack|web)\b'
        ]
        
        for pattern in tech_patterns:
            matches = re.findall(pattern, query_lower)
            terms.update(matches)
        
        # Method 2: Extract significant words and phrases
        # Remove stopwords
        stopwords = {'and', 'or', 'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                    'this', 'that', 'these', 'those', 'who', 'what', 'where', 'when', 'why', 'how',
                    'to', 'for', 'with', 'about', 'against', 'between', 'into', 'through',
                    'during', 'before', 'after', 'above', 'below', 'from', 'up', 'down',
                    'in', 'out', 'on', 'off', 'over', 'under', 'has', 'have', 'had',
                    'do', 'does', 'did', 'but', 'at', 'by', 'me', 'her', 'him', 'my',
                    'i', 'we', 'our', 'ours', 'you', 'your', 'yours', 'he', 'she', 'it',
                    'its', 'they', 'them', 'their', 'theirs', 'of', 'project', 'projects', 'worked',
                    'working', 'work', 'done', 'completed', 'person', 'people', 'list', 'show', 'tell'}
                    
        words = query_lower.split()
        filtered_words = [w for w in words if w not in stopwords and len(w) > 2]
        
        # Single words that might be significant
        for word in filtered_words:
            if len(word) > 3:  # Longer words are more likely significant
                terms.add(word)
        
        # Extract bigrams and trigrams (phrases of 2-3 words)
        for i in range(len(filtered_words)):
            # Bigrams
            if i < len(filtered_words) - 1:
                bigram = f"{filtered_words[i]} {filtered_words[i+1]}"
                terms.add(bigram)
                
            # Trigrams
            if i < len(filtered_words) - 2:
                trigram = f"{filtered_words[i]} {filtered_words[i+1]} {filtered_words[i+2]}"
                terms.add(trigram)
        
        # Filter out very common words that aren't technical terms
        filtered_terms = [term for term in terms if term not in stopwords]
        
        # Add the original query terms that might be relevant
        if len(filtered_terms) < 2:
            # If we didn't extract enough terms, add more from the original query
            additional_terms = [w for w in filtered_words if len(w) > 3]
            filtered_terms.extend(additional_terms)
            filtered_terms = list(set(filtered_terms))  # Deduplicate
        
        return filtered_terms

    def build_term_context_vectors(self, project_store: Dict) -> Dict[str, Any]:
        """
        Build context vectors for technical terms from the project corpus.
        
        This dynamically discovers relationships between terms based on how they're used
        in the project descriptions.
        
        Args:
            project_store: The project document store
            
        Returns:
            Dict: Mapping of terms to their context vectors
        """
        # Initialize containers
        term_docs = defaultdict(list)  # Term -> list of documents containing it
        doc_embeddings = {}  # Document ID -> embedding
        term_vectors = {}  # Term -> context vector
        
        # Process all projects to find term usage contexts
        doc_id = 0
        for project in project_store['all_projects']:
            # Create document text
            doc_text = f"{project['name']} {project['project_name']} {project['project_role']} {project['content']}"
            doc_lower = doc_text.lower()
            
            # Extract terms from document
            doc_terms = self.extract_key_terms(doc_lower)
            
            # Store document embedding
            doc_embedding = self.cached_embedding(doc_text)
            doc_embeddings[doc_id] = doc_embedding
            
            # Record which documents contain which terms
            for term in doc_terms:
                term_docs[term].append(doc_id)
            
            doc_id += 1
        
        # Calculate context vectors for each term
        for term, doc_ids in term_docs.items():
            if len(doc_ids) > 0:
                # Get embeddings of documents containing this term
                term_doc_embeddings = [doc_embeddings[doc_id] for doc_id in doc_ids]
                
                # Average the embeddings to get a context vector for this term
                if term_doc_embeddings:
                    # Convert all to numpy arrays
                    numpy_embeddings = [np.array(embedding) for embedding in term_doc_embeddings]
                    # Calculate mean vector
                    context_vector = np.mean(numpy_embeddings, axis=0)
                    term_vectors[term] = context_vector
        
        # Build co-occurrence matrix to find related terms
        cooccurrence = defaultdict(dict)
        for term1 in term_docs:
            docs1 = set(term_docs[term1])
            for term2 in term_docs:
                if term1 != term2:
                    docs2 = set(term_docs[term2])
                    # Calculate Jaccard similarity for co-occurrence
                    intersection = len(docs1 & docs2)
                    union = len(docs1 | docs2)
                    if union > 0:
                        cooccurrence[term1][term2] = intersection / union
        
        return {
            'term_vectors': term_vectors,
            'cooccurrence': dict(cooccurrence),
            'term_docs': dict(term_docs),
            'doc_count': doc_id
        }
    
    def expand_query_terms(self, query_terms: List[str], term_context_vectors: Dict, query_embedding) -> List[str]:
        """
        Expand query terms with related terms based on context vectors and co-occurrence.
        
        Args:
            query_terms: Original terms from the query
            term_context_vectors: Term context data from corpus
            query_embedding: Embedding of the full query
            
        Returns:
            List[str]: Expanded list of terms
        """
        term_vectors = term_context_vectors['term_vectors']
        cooccurrence = term_context_vectors['cooccurrence']
        
        expanded = set(query_terms)
        
        # Method 1: Add terms with high semantic similarity
        for term in query_terms:
            if term in term_vectors:
                term_vector = term_vectors[term]
                
                # Find similar terms based on vector similarity
                similarities = []
                for other_term, other_vector in term_vectors.items():
                    if other_term != term:
                        # Calculate cosine similarity
                        similarity = self.cosine_similarity_calc([term_vector], [other_vector])
                        similarity_value = float(similarity) if not hasattr(similarity, 'shape') else float(similarity[0][0])
                        similarities.append((other_term, similarity_value))
                
                # Add top similar terms
                similarities.sort(key=lambda x: x[1], reverse=True)
                for other_term, sim in similarities[:3]:
                    if sim > 0.75:  # High similarity threshold
                        expanded.add(other_term)
        
        # Method 2: Add terms with high co-occurrence
        for term in query_terms:
            if term in cooccurrence:
                related_terms = cooccurrence[term]
                # Add top co-occurring terms
                sorted_related = sorted(related_terms.items(), key=lambda x: x[1], reverse=True)
                for other_term, score in sorted_related[:3]:
                    if score > 0.3:  # Co-occurrence threshold
                        expanded.add(other_term)
        
        # Method 3: Check if any corpus terms are highly relevant to the query
        for term, vector in term_vectors.items():
            if term not in expanded:
                # Calculate relevance to query
                sim = self.cosine_similarity_calc([query_embedding], [vector])
                sim_value = float(sim) if not hasattr(sim, 'shape') else float(sim[0][0])
                if sim_value > 0.8:  # High relevance threshold
                    expanded.add(term)
        
        return list(expanded)   
    def determine_query_intent(self, query: str, query_embedding) -> Tuple[float, float]:
        """
        Determine if query is about a person's projects or about people on a project.
        
        Args:
            query: The user query
            query_embedding: Embedding of the query
            
        Returns:
            Tuple[float, float]: Scores for person-focused and project-focused intent
        """
        # Create embeddings for different query intents
        person_intent_embedding = self.cached_embedding(
            "Show me projects by a specific person. What projects did this person work on?"
        )
        project_intent_embedding = self.cached_embedding(
            "Who worked on a specific project? Show me people involved in this project."
        )
        
        # Compare query to intent embeddings
        person_intent_similarity = self.cosine_similarity_calc([query_embedding], [person_intent_embedding])
        person_intent_score = float(person_intent_similarity) if not hasattr(person_intent_similarity, 'shape') else float(person_intent_similarity[0][0])
        
        project_intent_similarity = self.cosine_similarity_calc([query_embedding], [project_intent_embedding])
        project_intent_score = float(project_intent_similarity) if not hasattr(project_intent_similarity, 'shape') else float(project_intent_similarity[0][0])
        
        return person_intent_score, project_intent_score   
    def name_similarity(self, name1, name2):
        """
        Calculate similarity specifically optimized for person names.
        This uses different weightings and specialized techniques for name comparison.
        
        Args:
            name1: First name to compare
            name2: Second name to compare
            
        Returns:
            float: Name similarity score (0-100)
        """
        # Handle empty strings
        if not name1 and not name2:  # Both empty
            return 100
        if not name1 or not name2:  # One empty
            return 0
        
        # Convert to lowercase for comparison
        name1 = name1.lower()
        name2 = name2.lower()
        
        # Calculate standard similarity metrics
        ratio = StringSimilarity.ratio(name1, name2)
        partial_ratio = StringSimilarity.partial_ratio(name1, name2)
        token_sort_ratio = StringSimilarity.token_sort_ratio(name1, name2)
        token_set_ratio = StringSimilarity.token_set_ratio(name1, name2)
        
        # Add specialized handling for names
        
        # 1. Initial letter match (important for names)
        initial_match = 10 if name1[0] == name2[0] else 0
        
        # 2. Handle name parts separately (first name, last name)
        name1_parts = name1.split()
        name2_parts = name2.split()
        
        # First part similarity (usually first name)
        first_part_similarity = 0
        if name1_parts and name2_parts:
            first_part_similarity = StringSimilarity.ratio(name1_parts[0], name2_parts[0])
        
        # Last part similarity (usually last name)
        last_part_similarity = 0
        if len(name1_parts) > 1 and len(name2_parts) > 1:
            last_part_similarity = StringSimilarity.ratio(name1_parts[-1], name2_parts[-1])
        
        # 3. Try to detect phonetic similarity
        phonetic_match = 0
        try:
            import jellyfish
            # Use Soundex for phonetic matching
            if jellyfish.soundex(name1) == jellyfish.soundex(name2):
                phonetic_match = 20
            
            # Use Metaphone for another phonetic algorithm
            if jellyfish.metaphone(name1) == jellyfish.metaphone(name2):
                phonetic_match += 15
                
            # Use Jaro-Winkler distance (specifically designed for names)
            jaro_winkler = jellyfish.jaro_winkler_similarity(name1, name2) * 15
            phonetic_match += jaro_winkler
        except ImportError:
            # No phonetic libraries available, use character set comparison instead
            common_chars = set(name1) & set(name2)
            total_chars = set(name1) | set(name2)
            if total_chars:
                phonetic_match = len(common_chars) / len(total_chars) * 20
        
        # Combine all signals with weights optimized for name matching
        weighted_score = (
            (ratio * 0.15) + 
            (partial_ratio * 0.25) + 
            (token_sort_ratio * 0.15) + 
            (token_set_ratio * 0.15) + 
            (initial_match * 0.05) +
            (first_part_similarity * 0.1) +
            (last_part_similarity * 0.1) +
            (phonetic_match * 0.05)
        )
        
        return weighted_score

    def initialize_name_matching(self):
        """
        Initialize all name matching components and ensure they're properly connected.
        This should be called during initialization of the class.
        """
        # Since we're using an instance method, no need to enhance StringSimilarity
        print("Enhanced name matching functionality initialized")
        return True
    def find_potential_person(self, query: str, project_store: Dict, query_embedding, term_context_vectors: Dict) -> Optional[str]:
        """
        Find potential person mentioned in query using improved case-insensitive matching.
        
        Args:
            query: The user query
            project_store: The project document store
            query_embedding: Embedding of the query
            term_context_vectors: Term context data from corpus
            
        Returns:
            Optional[str]: Potential person name if found with high confidence
        """
        query_lower = query.lower()
        all_persons = list(project_store['by_person'].keys())
        # Create case-insensitive mapping
        all_persons_lower = {person.lower(): person for person in all_persons}
        
        # Try direct matching of person names in query (case-insensitive)
        for person_lower, original_person in all_persons_lower.items():
            # Check for exact or partial matches (case-insensitive)
            if person_lower in query_lower:
                # Direct name mention
                return original_person
            
            # Check if the person's name is a substantial part of the query
            name_parts = person_lower.split()
            if len(name_parts) > 1:  # For people with first and last names
                # Check if full name is in query
                if person_lower in query_lower:
                    return original_person
                    
                # Check if both first and last name are in query (case-insensitive)
                query_words = set(query_lower.split())
                if all(part in query_words for part in name_parts):
                    return original_person
            
            # If only checking for a single name part, be more stringent
            elif person_lower in query_lower.split():
                # Single name exact match (like "John" in "Show John's projects")
                return original_person
        
        # Enhanced fuzzy matching for names
        potential_name_fragments = self.extract_name_fragments(query)
        name_matches = []
        
        # Match individual fragments against names (case-insensitive)
        for fragment in potential_name_fragments:
            fragment_lower = fragment.lower()
            for person_lower, original_person in all_persons_lower.items():
                name_parts = person_lower.split()
                
                # Check each part of the person's name and the whole name
                name_parts_to_check = name_parts + [person_lower]
                for part in name_parts_to_check:
                    # Use name_similarity for better name matching
                    similarity = self.StringSimilarity.name_similarity(fragment_lower, part)
                    
                    # Lower threshold to be more lenient
                    threshold = 60 if len(part) < 5 else 70
                    
                    if similarity > threshold:
                        name_matches.append((original_person, similarity))
        
        # Group scores by person and get highest score for each
        if name_matches:
            person_best_scores = {}
            for person, score in name_matches:
                if person not in person_best_scores or score > person_best_scores[person]:
                    person_best_scores[person] = score
            
            # Sort by score in descending order
            sorted_matches = sorted(person_best_scores.items(), key=lambda x: x[1], reverse=True)
            if sorted_matches:
                best_fuzzy_match = sorted_matches[0][0]
                best_fuzzy_score = sorted_matches[0][1]
                
                # More lenient confidence threshold for fuzzy matching
                if best_fuzzy_score > 60:
                    return best_fuzzy_match
        
        # Last attempt: semantic similarity (unchanged from original)
        best_score = 0
        best_match = None
        semantic_threshold = 0.65
        
        for person in all_persons:
            person_context = self.create_person_context(person, project_store['by_person'][person])
            person_embedding = self.cached_embedding(person_context)
            similarity = self.cosine_similarity_calc([query_embedding], [person_embedding])
            sim_value = float(similarity) if not hasattr(similarity, 'shape') else float(similarity[0][0])
            
            if sim_value > best_score and sim_value > semantic_threshold:
                best_score = sim_value
                best_match = person
        
        return best_match
    def find_name_boundaries(self, query: str, potential_name: str) -> tuple:
        """
        Find the precise boundaries of a potential name in a query string.
        This helps with extracting multi-word names from text and ensures
        we're replacing the right portion of text when enhancing queries.
        
        Args:
            query: The full query text (lowercase)
            potential_name: A potential name fragment found in the query (lowercase)
            
        Returns:
            tuple: (start_position, end_position) of the extended name
        """
        if not potential_name or not query:
            return (-1, -1)
            
        # Find initial position of the name fragment
        start_pos = query.find(potential_name)
        if start_pos == -1:
            return (-1, -1)
        
        end_pos = start_pos + len(potential_name)
        words = query.split()
        
        # Find which word the name starts in
        current_pos = 0
        start_word_idx = -1
        for i, word in enumerate(words):
            if current_pos <= start_pos < current_pos + len(word):
                start_word_idx = i
                break
            current_pos += len(word) + 1  # +1 for the space
        
        # Find which word the name ends in
        current_pos = 0
        end_word_idx = -1
        for i, word in enumerate(words):
            if current_pos < end_pos <= current_pos + len(word):
                end_word_idx = i
                break
            current_pos += len(word) + 1  # +1 for the space
        
        if start_word_idx == -1 or end_word_idx == -1:
            return (start_pos, end_pos)
        
        # Check if surrounding words might be part of the name
        extended_name_parts = []
        
        # Check words before the name
        if start_word_idx > 0:
            prev_word = words[start_word_idx-1]
            # Check if previous word starts with capital letter in original query
            # Since we're working with lowercase strings, we need a different approach
            # We'll check if this word is commonly a name prefix
            name_prefixes = ['mr', 'mrs', 'ms', 'dr', 'prof', 'sir', 'madam']
            if prev_word in name_prefixes or len(prev_word) > 1 and prev_word not in ['the', 'to', 'by', 'in', 'on', 'of', 'for', 'and', 'or']:
                extended_name_parts.append(prev_word)
        
        # Add the known name parts
        name_word_count = end_word_idx - start_word_idx + 1
        extended_name_parts.extend(words[start_word_idx:end_word_idx+1])
        
        # Check words after the name
        if end_word_idx < len(words)-1:
            next_word = words[end_word_idx+1]
            # Check if this might be a last name or suffix
            name_suffixes = ['jr', 'sr', 'ii', 'iii', 'iv', 'v']
            if next_word in name_suffixes or len(next_word) > 1 and next_word not in ['to', 'is', 'was', 'from', 'and', 'or', 'with', 'projects', 'skills']:
                # Only add if we don't already have too many words (unlikely to be a very long name)
                if name_word_count < 3:
                    extended_name_parts.append(next_word)
        
        # Calculate new boundaries based on extended name
        if not extended_name_parts:
            return (start_pos, end_pos)
            
        extended_name = " ".join(extended_name_parts)
        
        # Try to find the extended name in the query
        new_start_pos = query.find(extended_name)
        if new_start_pos != -1:
            new_end_pos = new_start_pos + len(extended_name)
            return (new_start_pos, new_end_pos)
        
        # If extended name can't be found directly (perhaps due to punctuation or other issues),
        # just return the original boundaries
        return (start_pos, end_pos)


    def enhance_string_similarity_for_names(self):
        """
        Add name-specific similarity methods to the StringSimilarity class.
        This method adds the name_similarity method which is optimized for comparing person names.
        
        Returns:
            StringSimilarity: The enhanced StringSimilarity class
        """
        @staticmethod
        def name_similarity(name1, name2):
            """
            Calculate similarity specifically optimized for person names.
            This uses different weightings and specialized techniques for name comparison.
            
            Args:
                name1: First name to compare
                name2: Second name to compare
                
            Returns:
                float: Name similarity score (0-100)
            """
            # Handle empty strings
            if not name1 and not name2:  # Both empty
                return 100
            if not name1 or not name2:  # One empty
                return 0
            
            # Convert to lowercase for comparison
            name1 = name1.lower()
            name2 = name2.lower()
            
            # Calculate standard similarity metrics
            ratio = StringSimilarity.ratio(name1, name2)
            partial_ratio = StringSimilarity.partial_ratio(name1, name2)
            token_sort_ratio = StringSimilarity.token_sort_ratio(name1, name2)
            token_set_ratio = StringSimilarity.token_set_ratio(name1, name2)
            
            # Add specialized handling for names
            
            # 1. Initial letter match (important for names)
            initial_match = 10 if name1[0] == name2[0] else 0
            
            # 2. Handle name parts separately (first name, last name)
            name1_parts = name1.split()
            name2_parts = name2.split()
            
            # First part similarity (usually first name)
            first_part_similarity = 0
            if name1_parts and name2_parts:
                first_part_similarity = StringSimilarity.ratio(name1_parts[0], name2_parts[0])
            
            # Last part similarity (usually last name)
            last_part_similarity = 0
            if len(name1_parts) > 1 and len(name2_parts) > 1:
                last_part_similarity = StringSimilarity.ratio(name1_parts[-1], name2_parts[-1])
            
            # 3. Try to detect phonetic similarity
            phonetic_match = 0
            try:
                import jellyfish
                # Use Soundex for phonetic matching
                if jellyfish.soundex(name1) == jellyfish.soundex(name2):
                    phonetic_match = 20
                
                # Use Metaphone for another phonetic algorithm
                if jellyfish.metaphone(name1) == jellyfish.metaphone(name2):
                    phonetic_match += 15
                    
                # Use Jaro-Winkler distance (specifically designed for names)
                jaro_winkler = jellyfish.jaro_winkler_similarity(name1, name2) * 15
                phonetic_match += jaro_winkler
            except ImportError:
                # No phonetic libraries available, use character set comparison instead
                common_chars = set(name1) & set(name2)
                total_chars = set(name1) | set(name2)
                if total_chars:
                    phonetic_match = len(common_chars) / len(total_chars) * 20
            
            # Combine all signals with weights optimized for name matching
            weighted_score = (
                (ratio * 0.15) + 
                (partial_ratio * 0.25) + 
                (token_sort_ratio * 0.15) + 
                (token_set_ratio * 0.15) + 
                (initial_match * 0.05) +
                (first_part_similarity * 0.1) +
                (last_part_similarity * 0.1) +
                (phonetic_match * 0.05)
            )
            
            return weighted_score
        
        # Add the method to the StringSimilarity class
        setattr(StringSimilarity, 'name_similarity', name_similarity)
        
        return StringSimilarity


    def extract_name_fragments(self, query: str) -> List[str]:
        """
        Extract potential name fragments from a query using n-grams and contextual clues.
        This is more effective for finding names in natural language text.
        
        Args:
            query: The user query
            
        Returns:
            List[str]: List of potential name fragments
        """
        query_lower = query.lower()
        fragments = []
        
        # 1. Extract single words that might be names
        for word in query_lower.split():
            if len(word) > 2 and word not in ['show', 'find', 'get', 'who', 'what', 'list', 'tell', 'give', 
                                            'project', 'projects', 'work', 'works', 'about', 'for', 'by', 
                                            'the', 'and', 'with', 'has', 'have', 'had', 'did', 'done']:
                fragments.append(word)
        
        # 2. Extract word pairs (bigrams)
        words = query_lower.split()
        for i in range(len(words) - 1):
            if len(words[i]) > 1 and len(words[i+1]) > 1:
                # Skip pairs with common stop words
                if words[i] not in ['the', 'and', 'of', 'to', 'in', 'for', 'with', 'by', 'about'] and \
                words[i+1] not in ['the', 'and', 'of', 'to', 'in', 'for', 'with', 'by', 'about']:
                    word_pair = f"{words[i]} {words[i+1]}"
                    fragments.append(word_pair)
        
        # 3. Extract word triplets (trigrams) for multi-part names
        for i in range(len(words) - 2):
            if all(len(words[i+j]) > 1 for j in range(3)):
                # Skip triplets with common stop words
                if not any(words[i+j] in ['the', 'and', 'of', 'to', 'in', 'for', 'with'] for j in range(3)):
                    triplet = f"{words[i]} {words[i+1]} {words[i+2]}"
                    fragments.append(triplet)
        
        # 4. Look for contextual patterns (words after "by", "of", "'s", etc.)
        name_indicators = [
        r'(?i)by\s+(\w+(?:\s+\w+){0,2})', 
        r'(?i)of\s+(\w+(?:\s+\w+){0,2})',
        r'(?i)from\s+(\w+(?:\s+\w+){0,2})',
        r'(?i)about\s+(\w+(?:\s+\w+){0,2})',
        r'(?i)(\w+)\'s',
        r'(?i)compare\s+(\w+(?:\s+\w+){0,2})\s+(?:and|with|vs|versus|to)',
        r'(?i)(?:and|with|vs|versus|to)\s+(\w+(?:\s+\w+){0,2})'
        ]
        
        for pattern in name_indicators:
            matches = re.findall(pattern, query)
            fragments.extend(matches)
        
        # 5. Filter duplicates and normalize
        unique_fragments = []
        for fragment in fragments:
            # Clean up fragments (remove punctuation, extra spaces)
            cleaned = re.sub(r'[^\w\s]', '', fragment).strip()
            if cleaned and cleaned not in unique_fragments:
                unique_fragments.append(cleaned)
        
        return unique_fragments


    # Integration function to ensure all the methods are properly connected
    def initialize_name_matching(self):
        """
        Initialize all name matching components and ensure they're properly connected.
        This should be called during initialization of the class.
        """
        # First enhance the StringSimilarity class with name_similarity method
        self.StringSimilarity = self.enhance_string_similarity_for_names()
        
        print("Enhanced name matching functionality initialized")
        
        # Return True to indicate success
        return True  
    def get_closest_name_match(self, query_name, all_names, threshold=70):
            """
            Get the closest name match using multiple string similarity methods.
            
            Args:
                query_name: The name to find matches for
                all_names: List of all available names
                threshold: Minimum similarity threshold (0-100)
                
            Returns:
                str or None: Best matching name or None if no good match
            """
            # If empty input, return None
            if not query_name or not all_names:
                return None
                
            # Use difflib get_close_matches first
            close_matches = difflib.get_close_matches(query_name, all_names, n=3, cutoff=0.6)
            
            if close_matches:
                return close_matches[0]  # Return the best match
            
            # If no matches with difflib, try our custom string similarity method
            matches = StringProcess.extract(query_name, all_names, limit=3)
            
            # Check if any match is above threshold
            best_match = None
            best_score = 0
            
            for match, score in matches:
                if score > threshold and score > best_score:
                    best_match = match
                    best_score = score
            
            return best_match
        
    def extract_potential_name(self, query):
            """
            Extract potential name from a query using heuristics.
            
            Args:
                query: The user query
                
            Returns:
                str or None: Potential name if found
            """
            # Common patterns for queries about a person
            patterns = [
        r"(?i)(?:what|show|find|list|get)\s+(?:projects|work)?\s*(?:by|for|of)\s+([A-Za-z][a-zA-Z]+(?:\s+[A-Za-z][a-zA-Z]+){0,2})",
        r"(?i)([A-Za-z][a-zA-Z]+(?:\s+[A-Za-z][a-zA-Z]+){0,2})(?:'s|\s+s)?\s+(?:projects|work|experience)",
        r"(?i)(?:who|what) is ([A-Za-z][a-zA-Z]+(?:\s+[A-Za-z][a-zA-Z]+){0,2})",
        r"(?i)tell me about ([A-Za-z][a-zA-Z]+(?:\s+[A-Za-z][a-zA-Z]+){0,2})",
        r"(?i)compare\s+([A-Za-z][a-zA-Z]+(?:\s+[A-Za-z][a-zA-Z]+){0,2})\s+(?:and|with|to|vs|versus)"
            ]
            
            for pattern in patterns:
                match = re.search(pattern, query)
                if match:
                    return match.group(1)
            
            # Look for capitalized words (potential names)
            words = query.split()
            name_candidates = []
            
            for i, word in enumerate(words):
                if len(word) > 0 and word[0].isupper() and len(word) > 2 and not word.lower() in ['show', 'find', 'get', 'who', 'what']:
                    # If this is a capitalized word, it might be the start of a name
                    name = word
                    
                    # Check if the next word is also capitalized (full name)
                    if i+1 < len(words) and words[i+1][0].isupper():
                        name += " " + words[i+1]
                        
                    name_candidates.append(name)
            
            # Return the longest name candidate (more likely to be a full name)
            if name_candidates:
                return max(name_candidates, key=len)
            
            return None

    def find_potential_project(self, query: str, project_store: Dict, query_embedding, term_context_vectors: Dict) -> Optional[str]:
        """
        Find potential project mentioned in query using advanced matching.
        
        Args:
            query: The user query
            project_store: The project document store
            query_embedding: Embedding of the query
            term_context_vectors: Term context data from corpus
            
        Returns:
            Optional[str]: Potential project name if found with high confidence
        """
        query_lower = query.lower()
        all_projects = list(project_store['by_project'].keys())
        all_projects_lower = {project.lower(): project for project in all_projects}
        # Extract query terms
        query_terms = self.extract_key_terms(query)
        
        # Try direct matching of project names in query
        best_match = None
        best_score = 0
        
        for project in all_projects:
            project_lower = project.lower()
            
            # Check for exact or partial matches
            if project_lower in query_lower:
                # Direct project mention
                return project
            
            # Check for term overlap
            project_terms = self.extract_key_terms(project_lower)
            term_overlap = len(set(query_terms) & set(project_terms))
            
            # Calculate semantic similarity between query and project context
            project_context = self.create_project_context(project, project_store['by_project'][project])
            project_embedding = self.cached_embedding(project_context)
            similarity = self.cosine_similarity_calc([query_embedding], [project_embedding])
            sim_value = float(similarity) if not hasattr(similarity, 'shape') else float(similarity[0][0])
            
            # Combined score (semantic + term overlap)
            combined_score = sim_value
            if term_overlap > 0:
                combined_score = sim_value * (1 + (term_overlap * 0.1))
            
            # Check if this project has the highest score
            if combined_score > best_score and combined_score > 0.75:  # High confidence threshold
                best_score = combined_score
                best_match = project
        
        return best_match
    
    def create_person_context(self, person_name: str, projects: List[Dict]) -> str:
        """
        Create a rich context description for a person based on their projects.
        
        Args:
            person_name: The person's name
            projects: List of the person's projects
            
        Returns:
            str: Contextual description of the person
        """
        context = f"Projects by {person_name}. {person_name} worked on "
        
        for i, project in enumerate(projects[:3]):  # Use top 3 projects
            if i > 0:
                context += ", and " if i == len(projects[:3])-1 else ", "
            context += f"{project['project_name']} as {project['project_role']}"
        
        # Add details from project content
        for project in projects[:2]:  # Add details from top 2 projects
            # Extract key sentences from content
            content = project['content']
            sentences = re.split(r'[.!?]', content)
            key_sentences = [s.strip() for s in sentences if len(s.strip()) > 30][:2]
            
            if key_sentences:
                context += ". " + ". ".join(key_sentences)
        
        return context
    
    def create_project_context(self, project_name: str, people: List[Dict]) -> str:
        """
        Create a rich context description for a project.
        
        Args:
            project_name: The project name
            people: List of people working on the project
            
        Returns:
            str: Contextual description of the project
        """
        context = f"Project: {project_name}. People who worked on {project_name}: "
        
        for i, person in enumerate(people[:3]):  # Use top 3 people
            if i > 0:
                context += ", and " if i == len(people[:3])-1 else ", "
            context += f"{person['name']} as {person['project_role']}"
        
        # Add details from project content
        for person in people[:2]:  # Add details from top 2 people
            # Extract key sentences from content
            content = person['content']
            sentences = re.split(r'[.!?]', content)
            key_sentences = [s.strip() for s in sentences if len(s.strip()) > 30][:2]
            
            if key_sentences:
                context += ". " + ". ".join(key_sentences)
        
        return context
    
    def calculate_term_relevance(self, query_terms: List[str], text: str, term_context_vectors: Dict) -> float:
        """
        Calculate relevance of text to query terms using both exact and semantic matching.
        
        Args:
            query_terms: Expanded query terms
            text: Text to analyze
            term_context_vectors: Term context data from corpus
            
        Returns:
            float: Term relevance score (0-1)
        """
        if not query_terms:
            return 0.0
        
        text_lower = text.lower()
        text_terms = self.extract_key_terms(text_lower)
        
        # Method 1: Direct term overlap
        direct_matches = set(query_terms) & set(text_terms)
        direct_score = len(direct_matches) / len(query_terms) if query_terms else 0
        
        # Method 2: Check co-occurrence patterns
        cooccurrence = term_context_vectors['cooccurrence']
        cooccur_score = 0
        
        for query_term in query_terms:
            if query_term in cooccurrence:
                for text_term in text_terms:
                    if text_term in cooccurrence[query_term]:
                        cooccur_score += cooccurrence[query_term][text_term]
        
        if query_terms and text_terms:
            cooccur_score /= len(query_terms) * len(text_terms)
        
        # Combined score with weighting
        return (direct_score * 0.7) + (cooccur_score * 0.3)


    def contextual_project_retrieval(self, query: str, llm=None, top_k: int = 5, similarity_threshold: float = 0.3):
        """
        Advanced contextual project retrieval .
        
        This approach:
        1. Uses embeddings to understand context beyond exact matches
        2. Dynamically discovers related terms from project corpus
        3. Leverages domain understanding through contextual embeddings
        4. Employs an ensemble approach combining multiple signals
     
        
        Args:
            query (str): The user query
            llm: Optional LLM for enhanced understanding
            top_k (int): Number of results to return
            similarity_threshold (float): Base threshold for semantic similarity
            
        Returns:
            dict: Results containing projects by person, by project name, and relevance
        """
        try:
            # Measure execution time
            start_time = time.time()
            
            # Step 1: Create project document store
            project_store = self.create_project_document_store()
            
            if not project_store['all_projects']:
                return {"results": [], "message": "No projects found in the database"}
            
            # Print available people for debugging
            print("Available people:", list(project_store['by_person'].keys()))
                
            # Step 2: Extract key terms from the query using both regex and embeddings
            query_terms = self.extract_key_terms(query)
            print(f"Extracted key terms from query: {query_terms}")
            
            # Step 3: Get query embedding for semantic similarity
            query_embedding = self.cached_embedding(query)
            
            # Step 4: Dynamically build a knowledge graph of related terms from the corpus
            # Use cached term context vectors if available
            if self.term_context_vectors_cache is None:
                self.term_context_vectors_cache = self.build_term_context_vectors(project_store)
            term_context_vectors = self.term_context_vectors_cache
            
            # Step 5: Find related terms for all query terms
            expanded_terms = self.expand_query_terms(query_terms, term_context_vectors, query_embedding)
            print(f"Expanded query terms: {expanded_terms}")
            
            # Step 6: Initialize result containers
            results = {
                "by_person": [],
                "by_project": [],
                "by_relevance": [],
          
            }
            
            # Step 7: Determine query intent with embeddings
            person_intent_score, project_intent_score = self.determine_query_intent(query, query_embedding)
     
            
            print(f"Intent scores - Person: {person_intent_score:.2f}, Project: {project_intent_score:.2f}")
            
         
            all_persons = list(project_store['by_person'].keys())
            preprocessed_query = self.name_matching(query, all_persons)
            potential_person = self.find_potential_person(preprocessed_query, project_store, query_embedding, term_context_vectors)
            potential_project = None
            
            #
            if not potential_person :
                potential_project = self.find_potential_project(query, project_store, query_embedding, term_context_vectors)
            
            # Step 9: Process based on intent and potential matches
            

            # Person-focused query - prioritize person-specific results
            if potential_person:
                print(f"Identified specific person: {potential_person}")
                
                # Get person's projects
                person_projects = project_store['by_person'].get(potential_person, [])
                
                # Score each project based on query relevance
                scored_projects = []
                for project in person_projects:
                    # Create project context for comparison
                    project_text = f"{project['project_name']} {project['project_role']} {project['content']}"
                    project_embedding = self.cached_embedding(project_text)
                    
                    # Calculate semantic similarity 
                    sem_similarity = self.cosine_similarity_calc([query_embedding], [project_embedding])
                    sem_sim_value = float(sem_similarity) if not hasattr(sem_similarity, 'shape') else float(sem_similarity[0][0])
                    
                    # Calculate term relevance
                    term_relevance = self.calculate_term_relevance(expanded_terms, project_text, term_context_vectors)
                    
                    # Combined score (weighted average)
                    combined_score = (sem_sim_value * 0.6) + (term_relevance * 0.4)
                    
                    # Add project regardless of threshold when person is specifically requested
                    project_copy = project.copy()
                    project_copy['similarity'] = combined_score
                    project_copy['semantic_similarity'] = sem_sim_value
                    project_copy['term_relevance'] = term_relevance
                    scored_projects.append(project_copy)
                
                # Sort by combined score
                scored_projects.sort(key=lambda x: x.get('similarity', 0), reverse=True)
                results["by_person"] = scored_projects
                
                # Clear other result types when specific person is requested
                results["by_project"] = []
                results["by_relevance"] = []
             
                
            # Project-focused query
            elif (project_intent_score > person_intent_score and 
               
                project_intent_score > 0.7) or potential_project:
                print(f"Query appears to be about a specific project (intent score: {project_intent_score:.2f})")
                
                if potential_project:
                    print(f"Identified potential project: {potential_project}")
                    
                    # Get people working on this project
                    project_people = project_store['by_project'].get(potential_project, [])
                    
                    # Score each person based on query relevance
                    scored_people = []
                    for person in project_people:
                        # Calculate relevance to query
                        person_text = f"{person['name']} {person['project_role']} {person['content']}"
                        person_embedding = self.cached_embedding(person_text)
                        
                        # Calculate semantic similarity
                        sem_similarity = self.cosine_similarity_calc([query_embedding], [person_embedding])
                        sem_sim_value = float(sem_similarity) if not hasattr(sem_similarity, 'shape') else float(sem_similarity[0][0])
                        
                        # Calculate term relevance 
                        term_relevance = self.calculate_term_relevance(expanded_terms, person_text, term_context_vectors)
                        
                        # Combined score
                        combined_score = (sem_sim_value * 0.6) + (term_relevance * 0.4)
                        
                        # Add to results if above threshold
                        if combined_score > similarity_threshold * 0.7:
                            person_copy = person.copy()
                            person_copy['similarity'] = combined_score
                            scored_people.append(person_copy)
                    
                    # If no people met the threshold, include at least some results
                    if not scored_people and project_people:
                        for person in project_people:
                            person_text = f"{person['name']} {person['project_role']} {person['content']}"
                            person_embedding = self.cached_embedding(person_text)
                            
                            sem_similarity = self.cosine_similarity_calc([query_embedding], [person_embedding])
                            sem_sim_value = float(sem_similarity) if not hasattr(sem_similarity, 'shape') else float(sem_similarity[0][0])
                            
                            term_relevance = self.calculate_term_relevance(expanded_terms, person_text, term_context_vectors)
                            
                            combined_score = (sem_sim_value * 0.6) + (term_relevance * 0.4)
                            
                            person_copy = person.copy()
                            person_copy['similarity'] = combined_score
                            scored_people.append(person_copy)
                    
                    # Sort by relevance
                    scored_people.sort(key=lambda x: x.get('similarity', 0), reverse=True)
                    results["by_project"] = scored_people
                else:
                    # No specific project identified, search across all projects
                    
                    # Score all projects by relevance to query
                    project_people = []
                    for project_name, people in project_store['by_project'].items():
                        # Calculate project relevance
                        project_context = self.create_project_context(project_name, people)
                        project_embedding = self.cached_embedding(project_context)
                        project_similarity = self.cosine_similarity_calc([query_embedding], [project_embedding])
                        project_sim_value = float(project_similarity) if not hasattr(project_similarity, 'shape') else float(project_similarity[0][0])
                        
                        # Check term relevance for project name
                        name_term_relevance = self.calculate_term_relevance(expanded_terms, project_name, term_context_vectors)
                        
                        # Combined project score
                        project_score = (project_sim_value * 0.7) + (name_term_relevance * 0.3)
                        
                        # Only consider if above threshold
                        if project_score > similarity_threshold * 0.6:
                            # Get people for this project
                            for person in people:
                                person_copy = person.copy()
                                person_copy['similarity'] = project_score
                                project_people.append(person_copy)
                    
                    # If no project people met the threshold, include the top matches anyway
                    if not project_people:
                        all_project_people = []
                        for project_name, people in project_store['by_project'].items():
                            project_context = self.create_project_context(project_name, people)
                            project_embedding = self.cached_embedding(project_context)
                            project_similarity = self.cosine_similarity_calc([query_embedding], [project_embedding])
                            project_sim_value = float(project_similarity) if not hasattr(project_similarity, 'shape') else float(project_similarity[0][0])
                            
                            name_term_relevance = self.calculate_term_relevance(expanded_terms, project_name, term_context_vectors)
                            project_score = (project_sim_value * 0.7) + (name_term_relevance * 0.3)
                            
                            for person in people:
                                person_copy = person.copy()
                                person_copy['similarity'] = project_score
                                all_project_people.append(person_copy)
                        
                        all_project_people.sort(key=lambda x: x.get('similarity', 0), reverse=True)
                        project_people = all_project_people[:top_k]
                    
                    # Sort by relevance
                    project_people.sort(key=lambda x: x.get('similarity', 0), reverse=True)
                    results["by_project"] = project_people[:top_k]
                    
            # Person-focused query (by intent but no specific person identified)
            elif (person_intent_score > project_intent_score and 
             
                person_intent_score > 0.7):
                print(f"Query appears to be about a person's projects (intent score: {person_intent_score:.2f})")
                
                # No specific person identified, search across all people
                
                # Score all persons by relevance to query
                person_projects = []
                for person_name, projects in project_store['by_person'].items():
                    # Calculate person relevance score
                    person_context = self.create_person_context(person_name, projects)
                    person_embedding = self.cached_embedding(person_context)
                    person_similarity = self.cosine_similarity_calc([query_embedding], [person_embedding])
                    person_sim_value = float(person_similarity) if not hasattr(person_similarity, 'shape') else float(person_similarity[0][0])
                    
                    # Only consider if above threshold
                    if person_sim_value > similarity_threshold * 0.6:
                        # Score each project for this person
                        for project in projects:
                            # Create project context
                            project_text = f"{project['project_name']} {project['project_role']} {project['content']}"
                            project_embedding = self.cached_embedding(project_text)
                            
                            # Calculate relevance scores
                            sem_similarity = self.cosine_similarity_calc([query_embedding], [project_embedding])
                            sem_sim_value = float(sem_similarity) if not hasattr(sem_similarity, 'shape') else float(sem_similarity[0][0])
                            
                            term_relevance = self.calculate_term_relevance(expanded_terms, project_text, term_context_vectors)
                            
                            # Combine scores with person relevance
                            combined_score = (sem_sim_value * 0.4) + (term_relevance * 0.3) + (person_sim_value * 0.3)
                            
                            # Add to results if above threshold
                            if combined_score > similarity_threshold * 0.7:
                                project_copy = project.copy()
                                project_copy['similarity'] = combined_score
                                person_projects.append(project_copy)
                
                # If no person projects met the threshold, include the top matches anyway
                if not person_projects:
                    # Collect all person-project combinations
                    all_person_projects = []
                    for person_name, projects in project_store['by_person'].items():
                        person_context = self.create_person_context(person_name, projects)
                        person_embedding = self.cached_embedding(person_context)
                        person_similarity = self.cosine_similarity_calc([query_embedding], [person_embedding])
                        person_sim_value = float(person_similarity) if not hasattr(person_similarity, 'shape') else float(person_similarity[0][0])
                        
                        for project in projects:
                            project_text = f"{project['project_name']} {project['project_role']} {project['content']}"
                            project_embedding = self.cached_embedding(project_text)
                            
                            sem_similarity = self.cosine_similarity_calc([query_embedding], [project_embedding])
                            sem_sim_value = float(sem_similarity) if not hasattr(sem_similarity, 'shape') else float(sem_similarity[0][0])
                            
                            term_relevance = self.calculate_term_relevance(expanded_terms, project_text, term_context_vectors)
                            
                            combined_score = (sem_sim_value * 0.4) + (term_relevance * 0.3) + (person_sim_value * 0.3)
                            
                            project_copy = project.copy()
                            project_copy['similarity'] = combined_score
                            all_person_projects.append(project_copy)
                    
                    # Take the top matches
                    all_person_projects.sort(key=lambda x: x.get('similarity', 0), reverse=True)
                    person_projects = all_person_projects[:top_k]
                
                # Sort by combined score
                person_projects.sort(key=lambda x: x.get('similarity', 0), reverse=True)
                results["by_person"] = person_projects[:top_k]
          
            else:
                print(f"Using general relevance search (person: {person_intent_score:.2f}, project: {project_intent_score:.2f})")
                
                # Search across all projects with expanded terms
                relevance_results = []
                all_scored_projects = []
                
                for project in project_store['all_projects']:
                    # Create comprehensive project text
                    project_text = f"{project['name']} worked on {project['project_name']} as {project['project_role']}. {project['content']}"
                    project_embedding = self.cached_embedding(project_text)
                    
                    # Calculate relevance scores
                    sem_similarity = self.cosine_similarity_calc([query_embedding], [project_embedding])
                    sem_sim_value = float(sem_similarity) if not hasattr(sem_similarity, 'shape') else float(sem_similarity[0][0])
                    
                    term_relevance = self.calculate_term_relevance(expanded_terms, project_text, term_context_vectors)
                    
                    # Combined score
                    combined_score = (sem_sim_value * 0.6) + (term_relevance * 0.4)
                    
                    # Store all scores for potential use
                    all_scored_projects.append((project, combined_score))
                    
                    # Add to results if above threshold
                    if combined_score > similarity_threshold * 0.7:
                        project_copy = project.copy()
                        project_copy['similarity'] = combined_score
                        relevance_results.append(project_copy)
                
                # If no projects met the threshold, include the top matches anyway
                if not relevance_results and all_scored_projects:
                    all_scored_projects.sort(key=lambda x: x[1], reverse=True)
                    top_projects = all_scored_projects[:top_k]
                    
                    for project, score in top_projects:
                        project_copy = project.copy()
                        project_copy['similarity'] = score
                        relevance_results.append(project_copy)
                
                # Sort by combined score
                relevance_results.sort(key=lambda x: x.get('similarity', 0), reverse=True)
                results["by_relevance"] = relevance_results[:top_k]
            
            # Post-processing: Ensure person-specific queries only show that person
            if potential_person:
                # Filter all results to ensure only the specified person is included
                if results["by_person"]:
                    results["by_person"] = [p for p in results["by_person"] if p["name"] == potential_person]
                # Clear other result types for person-specific queries
                results["by_project"] = []
                results["by_relevance"] = []
              
            

            
            # Log execution time
            end_time = time.time()
            print(f"Query processed in {end_time - start_time:.2f} seconds")
            
            # Print the results before returning
            print(f"Results: {results}")
            
            return {"results": results, "message": "Success"}
        
        except Exception as e:
            print(f"Error in contextual_project_retrieval: {e}")
            traceback.print_exc()
            return {"results": [], "message": f"Error: {str(e)}"}           

    def enhance_results_with_llm(self, query: str, results: Dict, llm) -> Dict:
        """
        Use Claude Haiku to enhance the relevance and ordering of results.
        
        Args:
            query: The original user query
            results: The results from contextual retrieval
            llm: The LLM to use for enhancement
            
        Returns:
            Dict: Enhanced results
        """
        try:
            # Only proceed if we have results to enhance
            if not results.get('results'):
                return results
                
            # Create a prompt to help Claude understand the query context
            context_prompt = """
            The user query is: "{query}"

            Analyze this query to understand what the user is looking for in terms of:
            1. Any specific technologies or skills mentioned (like AWS, Azure, etc.)
            2. Any specific person they're interested in
            3. Any specific project type they're looking for

            Respond with just a brief analysis of what's most important in this query, with no preamble.
            """
            
            # Fix: Replace dynamic parameter in template with string formatting
            context_prompt = context_prompt.format(query=query)
            
            # Get query analysis using a simpler approach that doesn't require template variables
            query_analysis_prompt = PromptTemplate(
                input_variables=[],
                template=context_prompt
            )
            chain = query_analysis_prompt | llm
            analysis = self.invoke_bedrock_with_retry(chain, {})
            query_analysis = analysis.content
            
            # Now rerank results based on this understanding
            for result_type in ['by_person', 'by_project', 'by_relevance']:
                if results['results'].get(result_type):
                    # Limit to a reasonable number to avoid token limits
                    items_to_rerank = results['results'][result_type][:10]
                    
                    # Create a representation of these results
                    items_json = json.dumps([{
                        'id': i,
                        'name': item.get('name', 'Unknown'),
                        'project_name': item.get('project_name', 'Unknown'),
                        'project_role': item.get('project_role', 'Unknown'),
                        # Include a truncated version of content to avoid token limits
                        'content_summary': item.get('content', '')[:500] + '...' if len(item.get('content', '')) > 500 else item.get('content', '')
                    } for i, item in enumerate(items_to_rerank)])
                    
                    # Create reranking prompt with string formatting instead of template variables
                    rerank_prompt_text = f"""
                    The user query is: "{query}"
                    
                    Query analysis: {query_analysis}
                    
                    Below are search results that may be relevant to this query:
                    {items_json}
                    
                    Please rerank these results in order of relevance to the user's query.
                    Return ONLY a comma-separated list of the ID numbers in the new order, most relevant first. 
                    For example: "2,5,1,3,4,0"
                    """
                    
                    # Get reranking
                    rerank_prompt = PromptTemplate(
                        input_variables=[],
                        template=rerank_prompt_text
                    )
                    chain = rerank_prompt | llm
                    reranking = self.invoke_bedrock_with_retry(chain, {})
                    reranking_text = reranking.content.strip()
                    
                    # Parse the reranking
                    try:
                        # Extract just the IDs if there's any additional text
                        reranking_text = re.search(r'[\d,\s]+', reranking_text).group(0)
                        new_order = [int(x.strip()) for x in reranking_text.split(',') if x.strip().isdigit()]
                        
                        # Validate the new order (ensure we have all original IDs)
                        if set(new_order) == set(range(len(items_to_rerank))):
                            # Reorder the results
                            reordered_items = [items_to_rerank[i] for i in new_order if i < len(items_to_rerank)]
                            
                            # Keep any items that weren't reranked
                            if len(reordered_items) < len(results['results'][result_type]):
                                reordered_items.extend(results['results'][result_type][len(reordered_items):])
                            
                            results['results'][result_type] = reordered_items
                    except Exception as e:
                        print(f"Error parsing reranking: {e}")
        
            return results

        except Exception as e:
            print(f"Error in enhance_results_with_llm: {e}")
            return results  # Return original results if enhancement fails

    # Add these methods to the ContextualProjectRetrieval class

    def format_project_results(self, results):
        """
        Format the results from contextual project retrieval into a readable output.
        
        Args:
            results (dict): Results from contextual_project_retrieval
            
        Returns:
            str: Formatted results as a string
        """
        if not results or not isinstance(results, dict):
            return "No results found or invalid results format."
        
        # Handle error messages
        if 'message' in results and results['message'] != "Success":
            return results['message']
        
        # Get the results data
        result_data = results.get('results', {})
        
        # Initialize output sections
        output_parts = []
        
        # Format person-specific results
        if 'by_person' in result_data and result_data['by_person']:
            output_parts.append("## Projects by Person")
            
            for project in result_data['by_person']:
                person_name = project.get('name', 'Unknown')
                project_name = project.get('project_name', 'Unknown Project')
                project_role = project.get('project_role', 'Unknown Role')
                similarity = project.get('similarity', 0)
                
                # Format the project details
                project_details = f"### {person_name}: {project_name}\n"
                project_details += f"**Role:** {project_role}\n"
                project_details += f"**Relevance:** {similarity*100:.1f}%\n\n"
                
                # Add project content
                content = project.get('content', '')
                if content:
                    # Extract key sections from content
                    sections = self.extract_content_sections(content)
                    
                    for section_title, section_text in sections.items():
                        if section_text:
                            project_details += f"**{section_title}:**\n{section_text}\n\n"
                
                output_parts.append(project_details)
        
        # Format project-specific results
        if 'by_project' in result_data and result_data['by_project']:
            output_parts.append("## People on Projects")
            
            # Group by project name
            projects_dict = {}
            for person in result_data['by_project']:
                project_name = person.get('project_name', 'Unknown Project')
                if project_name not in projects_dict:
                    projects_dict[project_name] = []
                projects_dict[project_name].append(person)
            
            # Format each project group
            for project_name, people in projects_dict.items():
                output_parts.append(f"### Project: {project_name}")
                
                for person in people:
                    person_name = person.get('name', 'Unknown')
                    project_role = person.get('project_role', 'Unknown Role')
                    similarity = person.get('similarity', 0)
                    
                    person_details = f"#### {person_name}\n"
                    person_details += f"**Role:** {project_role}\n"
                    person_details += f"**Relevance:** {similarity*100:.1f}%\n\n"
                    
                    # Add person's contribution to the project
                    content = person.get('content', '')
                    if content:
                        # Extract key sections from content
                        sections = self.extract_content_sections(content)
                        
                        for section_title, section_text in sections.items():
                            if section_text:
                                person_details += f"**{section_title}:**\n{section_text}\n\n"
                    
                    output_parts.append(person_details)
        
        # Format general relevance results
        if 'by_relevance' in result_data and result_data['by_relevance']:
            output_parts.append("## Most Relevant Projects")
            
            for project in result_data['by_relevance']:
                person_name = project.get('name', 'Unknown')
                project_name = project.get('project_name', 'Unknown Project')
                project_role = project.get('project_role', 'Unknown Role')
                similarity = project.get('similarity', 0)
                
                # Format the project details
                project_details = f"### {person_name} - {project_name}\n"
                project_details += f"**Role:** {project_role}\n"
                project_details += f"**Relevance:** {similarity*100:.1f}%\n\n"
                
                # Add project content
                content = project.get('content', '')
                if content:
                    # Extract key sections from content
                    sections = self.extract_content_sections(content)
                    
                    for section_title, section_text in sections.items():
                        if section_text:
                            project_details += f"**{section_title}:**\n{section_text}\n\n"
                
                output_parts.append(project_details)
        
        # If no results in any category
        if not output_parts:
            return "No relevant projects or people found for your query."
        
        return "\n".join(output_parts)

    def extract_content_sections(self, content):
        """
        Extract structured sections from project content.
        
        Args:
            content (str): Project content text
            
        Returns:
            dict: Dictionary of section title -> section text
        """
        sections = {}
        
        # Try to find common section headers
        section_patterns = [
            (r"Responsibilities?:(.*?)(?=Period:|Technologies:|Achievements:|$)", "Responsibilities"),
            (r"Period:(.*?)(?=Responsibilities:|Technologies:|Achievements:|$)", "Period"),
            (r"Technologies?:(.*?)(?=Period:|Responsibilities:|Achievements:|$)", "Technologies"),
            (r"Achievements?:(.*?)(?=Period:|Responsibilities:|Technologies:|$)", "Achievements"),
            (r"Description:(.*?)(?=Period:|Responsibilities:|Technologies:|Achievements:|$)", "Description")
        ]
        
        import re
        
        for pattern, section_name in section_patterns:
            matches = re.search(pattern, content, re.DOTALL)
            if matches:
                section_text = matches.group(1).strip()
                sections[section_name] = section_text
        
        # If no sections found, add entire content as description
        if not sections:
            sections["Description"] = content.strip()
        
        return sections

    def name_matching(self, query, all_persons):
        """
        Match a query to potential person names using string similarity.
        This is a wrapper around other name matching methods in the class.
        
        Args:
            query: The user query
            all_persons: List of all available person names
            
        Returns:
            str: The original query (preprocessed query was intended for further processing)
        """
        # Extract potential name from the query
        potential_name = self.extract_potential_name(query)
        
        if potential_name:
            # Try to find closest match if a potential name was found
            closest_match = self.get_closest_name_match(potential_name, all_persons)
            if closest_match:
                print(f"Potential name found in query: {potential_name} -> {closest_match}")
                return query
        
        # If no potential name was found, just return the original query
        return query  
    def query_projects_contextual(self, query: str, llm=None, top_k: int = 5):
        """
        High-level function to query projects using contextual retrieval.
        
        Args:
            query: The user query
            llm: Optional LLM for enhanced understanding (will initialize Claude Haiku if None)
            top_k: Number of results to return
            
        Returns:
            str: Formatted results
        """
        # Initialize Claude Haiku if no LLM is provided
        if llm is None:
            try:
                llm = ChatBedrock(
                    model_id="anthropic.claude-3-haiku-20240307-v1:0",
                    region_name=self.aws_region,
                    model_kwargs={
                        "temperature": 0.1,
                        "max_tokens": 1000,
                        "top_p": 0.9
                    }
                )
                print("Initialized Claude Haiku for enhanced project retrieval")
            except Exception as e:
                print(f"Could not initialize Claude Haiku: {e}. Proceeding without LLM enhancement.")
                llm = None
        
        # Get results using contextual retrieval
        results = self.contextual_project_retrieval(query, llm, top_k)
        if isinstance(results, dict) and results.get("is_comparison", False):
        # If it was a comparison query, the message already contains formatted results
            return results.get("message", "Comparison results unavailable.")
        # If we have LLM access, enhance the results
        if llm and results.get('results'):
            try:
                enhanced_results = self.enhance_results_with_llm(query, results, llm)
                results = enhanced_results
            except Exception as e:
                print(f"Error enhancing results with LLM: {e}")
        
        # Format and return results
        return self.format_project_results(results)
    
    def find_highest_role(roles):
        """
        Find the highest priority role from a list of roles.
        
        Args:
            roles: List of role titles
            
        Returns:
            str: The highest priority role
        """
        if not roles:
            return "Unknown Role"
        
        # Sort roles by priority and return the highest
        def get_role_priority(role):
            """
            Determine the priority of a role based on the specified hierarchy.
            
            Args:
                role: The role title (string)
                
            Returns:
                int: Priority value (lower is higher priority)
            """
            role_lower = role.lower()
            
            # Define role hierarchy with numeric priorities
            if 'architect' in role_lower:
                return 1
            elif 'lead' in role_lower and ('engineer' in role_lower or 'developer' in role_lower):
                return 2
            elif 'senior' in role_lower and ('engineer' in role_lower or 'developer' in role_lower):
                return 3
            elif ('engineer' in role_lower or 'developer' in role_lower) and not ('associate' in role_lower or 'junior' in role_lower):
                return 4
            elif 'associate' in role_lower and ('engineer' in role_lower or 'developer' in role_lower):
                return 5
            elif 'junior' in role_lower:
                return 6
            else:
                # For other roles, assign a lower priority
                return 10

        return sorted(roles, key=get_role_priority)[0]

    def get_latest_roles(people_names, project_store):
        """
        Get the latest/highest role for each person.
        
        Args:
            people_names: List of people to compare
            project_store: The project document store
            
        Returns:
            Dict[str, str]: Dictionary mapping person name to their highest role
        """
        latest_roles = {}
        
        for person_name in people_names:
            person_projects = project_store['by_person'].get(person_name, [])
            all_roles = []
            
            for project in person_projects:
                role = project['project_role'].strip()
                if role:
                    all_roles.append(role)
            
            # Find the highest priority role
            highest_role = ContextualProjectRetrieval.find_highest_role(all_roles)
            latest_roles[person_name] = highest_role
        
        return latest_roles

    def format_comparison_results_tabular(self, comparison_results: Dict, query: str) -> str:
        """
        Format comparison results in a tabular format using markdown.
        - Prioritizes roles in hierarchy: Architect > Lead Engineer > Senior Engineer > Engineer > Associate Engineer
        - Displays only latest/highest role for each person
        - Removes the project role table with all roles
        
        Args:
            comparison_results: The comparison results
            query: The original user query
            
        Returns:
            str: Formatted tabular comparison output
        """
        people_names = [person["name"] for person in comparison_results["people"]]
        
        # Limit to 4 people maximum
        if len(people_names) > 4:
            people_names = people_names[:4]
            # Filter comparison results to include only these people
            for key in ["skill_comparison", "project_comparison", "availability_comparison"]:
                if key in comparison_results:
                    if isinstance(comparison_results[key], dict):
                        for subkey in list(comparison_results[key].keys()):
                            if isinstance(comparison_results[key][subkey], dict):
                                comparison_results[key][subkey] = {
                                    k: v for k, v in comparison_results[key][subkey].items() 
                                    if k in people_names
                                }
        
        # Get query embedding for analysis
        query_embedding = self.cached_embedding(query)
        
        # Analyze query intent
        query_intent = self.analyze_query_intent(query, query_embedding)
        
        # Create the project store for role analysis
        project_store = self.create_project_document_store()
        
        # Get latest/highest roles for each person
        latest_roles = ContextualProjectRetrieval.get_latest_roles(people_names, project_store)
        
        formatted_output = []
        
        # Generate title
        if len(people_names) == 2:
            formatted_output.append(f"# Comparison: {people_names[0]} vs. {people_names[1]}")
        else:
            formatted_output.append(f"# Comparison of {len(people_names)} People")
            formatted_output.append(people_names[0] + " vs. " + " vs. ".join(people_names[1:]))
        
        # If query indicates skills focus, add it to the title
        skills_focus = query_intent["skills_focus"]
        if skills_focus:
            formatted_output.append(f"\nFocus: {', '.join(skills_focus)}")
        
        # Add detected focus areas
        project_types = query_intent["project_types"]
        if project_types:
            formatted_output.append(f"Project Types: {', '.join(project_types)}")
        
        # Add current roles section (replaces the project roles table)
        formatted_output.append("\n## Current Roles\n")
        
        # Create current roles table
        roles_table = ["| Person | Current Role |",
                    "| --- | --- |"]
        
        # Sort people by role priority
        sorted_people = sorted(people_names, key=lambda person: ContextualProjectRetrieval.get_role_priority(latest_roles.get(person, "Unknown")))
        
        for person in sorted_people:
            role = latest_roles.get(person, "Unknown Role")
            roles_table.append(f"| {person} | {role} |")
        
        # Add roles table
        formatted_output.extend(roles_table)
        
        # Add query-focused skills first if mentioned in query
        if skills_focus and comparison_results["focus_areas"]["skills"]:
            formatted_output.append("\n## Query-Focused Skills\n")
            
            # Create skill comparison table header
            skill_table = ["| Skill | " + " | ".join(people_names) + " |",
                        "| --- | " + " | ".join(["---"] * len(people_names)) + " |"]
            
            # For each skill focus, find matching skills in the comparison results
            for focus_skill in skills_focus:
                matching_skills = []
                
                # Find exact and partial matches
                for skill in comparison_results["skill_comparison"].keys():
                    # Check for matches in both directions
                    if focus_skill.lower() in skill.lower() or skill.lower() in focus_skill.lower():
                        matching_skills.append(skill)
                
                # If no exact matches, try fuzzy matching
                if not matching_skills:
                    for skill in comparison_results["skill_comparison"].keys():
                        # Simple fuzzy matching using character overlap
                        focus_chars = set(focus_skill.lower())
                        skill_chars = set(skill.lower())
                        overlap = len(focus_chars & skill_chars) / len(focus_chars | skill_chars)
                        
                        if overlap > 0.6:  # 60% character overlap threshold
                            matching_skills.append(skill)
                
                # Add rows for matching skills
                for skill in matching_skills:
                    row = [f"**{skill}**"]  # Bold to highlight query-focused skills
                    for person in people_names:
                        confidence = comparison_results["skill_comparison"][skill].get(person, 0)
                        if confidence > 0.9:
                            row.append("Expert")
                        elif confidence > 0.8:
                            row.append("Advanced")
                        elif confidence > 0.7:
                            row.append("Proficient")
                        elif confidence > 0.6:
                            row.append("Familiar")
                        else:
                            row.append("-")
                    
                    skill_table.append("| " + " | ".join(row) + " |")
            
            # Only add the table if we found matching skills
            if len(skill_table) > 2:
                formatted_output.extend(skill_table)
            else:
                # Rest of the section unchanged...
                alternative_skills = []
                
                # Look for alternative skills that might be relevant
                for focus_skill in skills_focus:
                    focus_embedding = self.cached_embedding(focus_skill)
                    
                    for skill in list(comparison_results["skill_comparison"].keys())[:20]:  # Limit to first 20 for performance
                        try:
                            skill_embedding = self.cached_embedding(skill)
                            similarity = self.cosine_similarity_calc([focus_embedding], [skill_embedding])
                            sim_value = float(similarity) if not hasattr(similarity, 'shape') else float(similarity[0][0])
                            
                            if sim_value > 0.7:  # Semantic similarity threshold
                                alternative_skills.append((skill, sim_value))
                        except Exception as e:
                            print(f"Error comparing skill embeddings: {e}")
                
                # Sort by similarity and take top 3
                alternative_skills.sort(key=lambda x: x[1], reverse=True)
                
                if alternative_skills:
                    formatted_output.append(f"No exact matches found for '{', '.join(skills_focus)}', but found these related skills:")
                    
                    # Create table for alternative skills
                    alt_skill_table = ["| Skill | " + " | ".join(people_names) + " |",
                                    "| --- | " + " | ".join(["---"] * len(people_names)) + " |"]
                    
                    for skill, _ in alternative_skills[:3]:
                        row = [skill]
                        for person in people_names:
                            confidence = comparison_results["skill_comparison"][skill].get(person, 0)
                            if confidence > 0.9:
                                row.append("Expert")
                            elif confidence > 0.8:
                                row.append("Advanced")
                            elif confidence > 0.7:
                                row.append("Proficient")
                            elif confidence > 0.6:
                                row.append("Familiar")
                            else:
                                row.append("-")
                        
                        alt_skill_table.append("| " + " | ".join(row) + " |")
                    
                    formatted_output.extend(alt_skill_table)
                else:
                    formatted_output.append(f"No skills found matching '{', '.join(skills_focus)}'.")
        
        # Add general skills comparison
        if comparison_results["focus_areas"]["skills"]:
            formatted_output.append("\n## Skills Comparison\n")
            
            # Create skill comparison table header
            skill_table = ["| Skill | " + " | ".join(people_names) + " |",
                        "| --- | " + " | ".join(["---"] * len(people_names)) + " |"]
            
            # Get top skills (with confidence > 0.7 for any person)
            relevant_skills = [
                skill for skill, person_confidences in comparison_results["skill_comparison"].items()
                if any(confidence > 0.7 for confidence in person_confidences.values())
            ]
            
            # Sort skills by average confidence
            sorted_skills = sorted(
                relevant_skills,
                key=lambda skill: sum(comparison_results["skill_comparison"][skill].values()) / len(people_names),
                reverse=True
            )
            
            # Filter out skills already shown in query-focused section
            if skills_focus:
                filtered_skills = []
                for skill in sorted_skills:
                    already_shown = False
                    for focus_skill in skills_focus:
                        if focus_skill.lower() in skill.lower() or skill.lower() in focus_skill.lower():
                            already_shown = True
                            break
                    if not already_shown:
                        filtered_skills.append(skill)
                sorted_skills = filtered_skills
            
            # Create rows for skills
            for skill in sorted_skills[:15]:  # Limit to top 15 skills
                row = [skill]
                for person in people_names:
                    confidence = comparison_results["skill_comparison"][skill].get(person, 0)
                    if confidence > 0.9:
                        row.append("Expert")
                    elif confidence > 0.8:
                        row.append("Advanced")
                    elif confidence > 0.7:
                        row.append("Proficient")
                    elif confidence > 0.6:
                        row.append("Familiar")
                    else:
                        row.append("-")
                
                skill_table.append("| " + " | ".join(row) + " |")
            
            # Add skill table if we have skills to show
            if len(skill_table) > 2:
                formatted_output.extend(skill_table)
        
        # Add project comparison table if focus area includes projects
        if comparison_results["focus_areas"]["projects"]:
            formatted_output.append("\n## Project Experience\n")
            
            # Create project metrics table
            project_table = ["| Metric | " + " | ".join(people_names) + " |",
                        "| --- | " + " | ".join(["---"] * len(people_names)) + " |"]
            
            # Add project count row
            project_counts = []
            for person in people_names:
                count = comparison_results["project_comparison"]["project_count"].get(person, 0)
                project_counts.append(str(count))
            project_table.append("| Total Projects | " + " | ".join(project_counts) + " |")
            
            # Add query-specific project counts if project types are mentioned
            if project_types:
                for project_type in project_types:
                    type_counts = []
                    for person in people_names:
                        # Count projects of this type for each person
                        count = 0
                        for p in comparison_results["people"]:
                            if p["name"] == person:
                                for project in p.get("top_projects", []):
                                    content = (project.get("content", "") + " " + project.get("name", "")).lower()
                                    if project_type.lower() in content:
                                        count += 1
                        type_counts.append(str(count))
                    
                    project_table.append(f"| {project_type.title()} Projects | " + " | ".join(type_counts) + " |")
            
            # Extract and add average project duration
            durations = []
            for person in people_names:
                avail = comparison_results["availability_comparison"].get(person, {})
                duration = avail.get("average_project_duration_months", 0)
                durations.append(f"{duration:.1f} months")
            project_table.append("| Avg Project Duration | " + " | ".join(durations) + " |")
            
            # Add project table
            formatted_output.extend(project_table)
            
            # Note: Project roles table removed as per requirement
            
            # Add unique projects table
            unique_projects = comparison_results["project_comparison"]["unique_projects"]
            if any(projects for person, projects in unique_projects.items() if person in people_names):
                formatted_output.append("\n## Unique Projects\n")
                
                unique_table = ["| Person | Unique Projects |",
                            "| --- | --- |"]
                
                for person in people_names:
                    projects = unique_projects.get(person, [])
                    
                    # Prioritize projects matching query focus
                    if projects and (skills_focus or project_types):
                        # Sort projects to show query-relevant ones first
                        query_relevant = []
                        other_projects = []
                        
                        focus_terms = skills_focus + project_types
                        
                        for proj in projects[:10]:  # Limit to first 10 to avoid excessive processing
                            if any(term.lower() in proj.lower() for term in focus_terms):
                                query_relevant.append(f"**{proj}**")  # Bold to highlight
                            else:
                                other_projects.append(proj)
                        
                        # Combine, showing relevant ones first
                        sorted_projects = query_relevant + other_projects
                        
                        if sorted_projects:
                            unique_table.append(f"| {person} | {', '.join(sorted_projects[:5])}" + (", ..." if len(sorted_projects) > 5 else "") + " |")
                        else:
                            unique_table.append(f"| {person} | None |")
                    elif projects:
                        unique_table.append(f"| {person} | {', '.join(projects[:5])}" + (", ..." if len(projects) > 5 else "") + " |")
                    else:
                        unique_table.append(f"| {person} | None |")
                
                formatted_output.extend(unique_table)
        
        # The rest of the function remains mostly unchanged...
        
        # Add technologies comparison
        formatted_output.append("\n## Technologies & Tools\n")
        
        # Gather technologies from skills and projects
        all_technologies = set()
        person_technologies = {person: set() for person in people_names}
        
        # Standard technology keywords to look for - expanded to catch more technologies
        technology_keywords = [
            "aws", "azure", "gcp", "cloud", "kubernetes", "docker", "terraform", 
            "jenkins", "gitlab", "github", "python", "java", "javascript", "c#",
            "react", "angular", "node", "spring", "django", "flask", "sql", "nosql",
            "mongodb", "postgresql", "mysql", "oracle", "redis", "kafka", "hadoop",
            "spark", "elasticsearch", "linux", "windows", "ios", "android", "mobile",
            # Add more technologies
            "typescript", "go", "ruby", "php", "rust", "swift", "kotlin", "objective-c",
            "vue", "express", "asp.net", "laravel", "symfony", "rails", "jquery",
            "dynamodb", "cassandra", "couchdb", "firebase", "neo4j", "graphql",
            "docker-compose", "swarm", "istio", "helm", "openshift", "lambda",
            "ec2", "s3", "ecs", "fargate", "eks", "rds", "aurora", "redshift",
            "cloudformation", "pulumi", "ansible", "puppet", "chef", "datadog",
            "prometheus", "grafana", "elk", "oauth", "jwt", "rest", "soap", "grpc"
        ]
        
        for person in people_names:
            # From skills
            for skill in comparison_results["skill_comparison"]:
                if person in comparison_results["skill_comparison"][skill] and comparison_results["skill_comparison"][skill][person] > 0.6:
                    skill_lower = skill.lower()
                    if any(tech in skill_lower for tech in technology_keywords):
                        person_technologies[person].add(skill_lower)
                        all_technologies.add(skill_lower)
            
            # From projects
            for p in comparison_results["people"]:
                if p["name"] == person:
                    for project in p.get("top_projects", []):
                        content = (project.get("content", "") + " " + project.get("role", "") + " " + project.get("name", "")).lower()
                        
                        for tech in technology_keywords:
                            if tech in content:
                                person_technologies[person].add(tech)
                                all_technologies.add(tech)
        
        # Add query-focused technologies first if query mentions technologies
        query_techs = set()
        if skills_focus:
            for focus in skills_focus:
                for tech in all_technologies:
                    if focus.lower() in tech or tech in focus.lower():
                        query_techs.add(tech)
        
        if query_techs:
            formatted_output.append("### Query-Focused Technologies\n")
            
            # Create technologies table header
            tech_table = ["| Technology | " + " | ".join(people_names) + " |",
                        "| --- | " + " | ".join(["---"] * len(people_names)) + " |"]
            
            # Add rows for query-related technologies
            for tech in sorted(query_techs):
                row = [tech.title()]  # Capitalize technology name
                for person in people_names:
                    has_tech = tech in person_technologies[person]
                    row.append("✓" if has_tech else "✗")
                
                tech_table.append("| " + " | ".join(row) + " |")
            
            formatted_output.extend(tech_table)
            
            # Remove query technologies from all technologies to avoid duplication
            all_technologies -= query_techs
        
        # Create general technologies table
        tech_table = ["| Technology | " + " | ".join(people_names) + " |",
                    "| --- | " + " | ".join(["---"] * len(people_names)) + " |"]
        
        # Sort technologies alphabetically
        for tech in sorted(all_technologies):
            row = [tech.title()]  # Capitalize technology name
            for person in people_names:
                has_tech = tech in person_technologies[person]
                row.append("✓" if has_tech else "✗")
            
            tech_table.append("| " + " | ".join(row) + " |")
        
        if len(tech_table) > 2:  # Only add if we have technologies
            if query_techs:
                formatted_output.append("\n### Other Technologies\n")
            formatted_output.extend(tech_table)
        
        # Add availability comparison table if focus area includes availability
        if comparison_results["focus_areas"]["availability"]:
            formatted_output.append("\n## Availability\n")
            
            # Create availability table
            avail_table = ["| Metric | " + " | ".join(people_names) + " |",
                        "| --- | " + " | ".join(["---"] * len(people_names)) + " |"]
            
            # Add rows for each availability metric
            metrics = [
                ("Current Project", lambda p, a: a.get("current_project") or "-"),
                ("Current Role", lambda p, a: latest_roles.get(p, "-")),  # Use latest role instead of a.get("current_role")
                ("Expected End Date", lambda p, a: a.get("project_end_date") or "-"),
                ("Available", lambda p, a: "Yes" if a.get("is_available", False) else "No"),
                ("Availability Confidence", lambda p, a: f"{a.get('availability_confidence', 0)*100:.0f}%")
            ]
            
            for metric_name, value_func in metrics:
                row = [metric_name]
                for person in people_names:
                    avail = comparison_results["availability_comparison"].get(person, {})
                    row.append(value_func(person, avail))
                
                avail_table.append("| " + " | ".join(row) + " |")
            
            # Add availability table
            formatted_output.extend(avail_table)
        
        # Continue with common skills and ranking sections unchanged...
        
        return "\n".join(formatted_output)

    def detect_comparison_intent(self, query: str, query_embedding) -> float:
        """
        Determine if query is a comparison between multiple people.
        
        Args:
            query: The user query
            query_embedding: Embedding of the query
            
        Returns:
            float: Score for comparison intent
        """
        # Create embeddings for comparison query intent
        comparison_intent_embedding = self.cached_embedding(
            "Compare these people. Show differences between these individuals. Who has better skills between these people?"
        )
        
        # Compare query to comparison intent embedding
        comparison_intent_similarity = self.cosine_similarity_calc([query_embedding], [comparison_intent_embedding])
        comparison_intent_score = float(comparison_intent_similarity) if not hasattr(comparison_intent_similarity, 'shape') else float(comparison_intent_similarity[0][0])
        
        # Use regex patterns to boost detection
        comparison_patterns = [
            r"(?i)compare\s+(\w+)\s+(?:and|vs|versus|with|to)\s+(\w+)",
            r"(?i)difference\s+between\s+(\w+)\s+(?:and|vs|versus)\s+(\w+)",
            r"compare\s+(\w+)\s+(?:and|vs|versus|with|to)\s+(\w+)",
            r"difference\s+between\s+(\w+)\s+(?:and|vs|versus)\s+(\w+)",
            r"(?:who|which)\s+(?:is|are)\s+(?:better|more experienced|stronger)",
            r"(?:availability|schedule)\s+(?:of|for)\s+(\w+)\s+(?:and|vs|versus|with|,)\s+(\w+)",
            r"(\w+)\s+(?:vs|versus)\s+(\w+)"
        ]
        
        pattern_match_boost = 0.0
        for pattern in comparison_patterns:
            if re.search(pattern, query.lower()):
                pattern_match_boost = 0.3
                break
        
        return comparison_intent_score + pattern_match_boost

    def identify_multiple_people(self, query: str, project_store: Dict) -> List[str]:
        """
        Identify multiple people mentioned in a comparison query with improved case-insensitive matching.
        
        Args:
            query: The user query
            project_store: The project document store
            
        Returns:
            List[str]: List of identified people names (empty list if none found)
        """
        # Always initialize with an empty list, not None
        identified_people = []
        
        # Skip processing if inputs are invalid
        if not query or not project_store or 'by_person' not in project_store:
            return identified_people
        
        query_lower = query.lower()
        all_persons = list(project_store['by_person'].keys())
        
        # Create case-insensitive mapping of names (this is key to the solution)
        all_persons_lower = {person.lower(): person for person in all_persons}
        
        print(f"Looking for names in query: {query}")
        print(f"Available people: {all_persons}")
        
        # Step 1: Check for direct name mentions with improved matching
        for person_lower, original_person in all_persons_lower.items():
            # Check for exact or partial matches (case-insensitive)
            if person_lower in query_lower:
                print(f"Found exact match: {original_person}")
                identified_people.append(original_person)
                continue
            
            # Check first name and last name separately
            name_parts = person_lower.split()
            if len(name_parts) > 1:  # For people with first and last names
                # Check if both first and last name are in query (case-insensitive)
                if all(part in query_lower for part in name_parts):
                    print(f"Found match by name parts: {original_person}")
                    identified_people.append(original_person)
                    continue
            
            # For single name parts, be more stringent
            elif person_lower in query_lower.split():
                print(f"Found match by word: {original_person}")
                identified_people.append(original_person)
                continue
        
        # Step 2: If exact matches aren't found, try fuzzy matching
        if len(identified_people) < 2:
            # Extract potential name fragments using improved technique
            potential_names = []
            
            # Look for capitalized words that might be names
            for word in query.split():
                if word and word[0].isupper() and len(word) > 2:
                    potential_names.append(word.lower())
            
            # Look for pairs of capitalized words (first+last name)
            words = query.split()
            for i in range(len(words) - 1):
                if words[i] and words[i+1] and words[i][0].isupper() and words[i+1][0].isupper():
                    potential_names.append(f"{words[i].lower()} {words[i+1].lower()}")
            
            print(f"Potential name fragments: {potential_names}")
            
            # Try partial matches using string similarity
            for potential_name in potential_names:
                best_match = None
                best_score = 0
                threshold = 70  # Lower threshold to improve matching
                
                for person_lower, original_person in all_persons_lower.items():
                    if original_person in identified_people:
                        continue  # Skip already identified people
                    
                    # Use string similarity for matching
                    try:
                        ratio = self.StringSimilarity.ratio(potential_name, person_lower)
                        partial_ratio = self.StringSimilarity.partial_ratio(potential_name, person_lower)
                        
                        # Use lower threshold for partial matching
                        combined_score = (ratio * 0.4) + (partial_ratio * 0.6)
                        
                        if combined_score > best_score and combined_score > threshold:
                            best_score = combined_score
                            best_match = original_person
                    except Exception as e:
                        print(f"Error in string similarity: {e}")
                        continue
                
                if best_match:
                    print(f"Found fuzzy match: {best_match} (score: {best_score})")
                    identified_people.append(best_match)
        
        # Step 3: If still not enough matches, look for people mentioned in context
        if len(identified_people) < 2:
            # Look for specific comparison patterns
            comparison_patterns = [
                r"(?:compare|vs|versus)\s+([A-Za-z\s]+)\s+(?:and|with|to)\s+([A-Za-z\s]+)",
                r"([A-Za-z\s]+)\s+(?:vs|versus)\s+([A-Za-z\s]+)",
                r"(?:difference|differences)\s+between\s+([A-Za-z\s]+)\s+and\s+([A-Za-z\s]+)"
            ]
            
            import re
            for pattern in comparison_patterns:
                matches = re.search(pattern, query, re.IGNORECASE)
                if matches:
                    name1 = matches.group(1).strip()
                    name2 = matches.group(2).strip()
                    
                    print(f"Found comparison pattern: '{name1}' vs '{name2}'")
                    
                    # Match these names to the database
                    for extracted_name in [name1, name2]:
                        extracted_lower = extracted_name.lower()
                        
                        # Skip if already identified
                        if any(person for person in identified_people if person.lower() == extracted_lower):
                            continue
                        
                        # Find best matching person
                        best_match = None
                        best_score = 0
                        
                        for person_lower, original_person in all_persons_lower.items():
                            if original_person in identified_people:
                                continue
                            
                            # Try exact match first
                            if extracted_lower == person_lower:
                                best_match = original_person
                                break
                            
                            # Try partial match with higher threshold
                            try:
                                # More weight on partial_ratio for name matching
                                similarity = self.StringSimilarity.partial_ratio(extracted_lower, person_lower)
                                
                                if similarity > best_score and similarity > 60:  # Lower threshold
                                    best_score = similarity
                                    best_match = original_person
                            except Exception as e:
                                print(f"Error in string similarity for names: {e}")
                                continue
                        
                        if best_match and best_match not in identified_people:
                            print(f"Found pattern match: {best_match}")
                            identified_people.append(best_match)
                    
                    break  # Exit after first successful pattern match
        
        # Always return a list (may be empty, but never None)
        print(f"Final identified people: {identified_people}")
        return identified_people[:5]  # Limit to 5 people to avoid excessive comparisons


    def handle_comparison_query(self, query: str, llm=None, top_k: int = 5):
        """
        Modified handle_comparison_query function with improved error handling.
        """
        try:
            # Measure execution time
            start_time = time.time()
            
            # Create project document store
            project_store = self.create_project_document_store()
            
            if not project_store or not project_store.get('all_projects'):
                return "No projects found in the database to compare."
            
            # Get query embedding for semantic analysis
            query_embedding = self.cached_embedding(query)
            
            # Identify people to compare with improved handling
            people_names = self.identify_multiple_people(query, project_store)
            
            # Safety check - ensure people_names is a list and not None
            if not isinstance(people_names, list):
                people_names = []
                
            # Check if we have enough people to compare
            if len(people_names) < 2:
                return f"Please specify at least two people to compare. I couldn't reliably identify multiple people in your query: '{query}'. Try mentioning their full names directly."
            
            print(f"Comparing people: {people_names}")
            
            # Limit to 4 people maximum
            if len(people_names) > 4:
                people_names = people_names[:4]
            
            # Perform comparison
            comparison_results = self.compare_people(people_names, query, project_store, query_embedding)
            
            # Format results
            formatted_results = self.format_comparison_results_tabular(comparison_results, query)
            
            # Log execution time
            end_time = time.time()
            print(f"Comparison processed in {end_time - start_time:.2f} seconds")
            
            return formatted_results
        
        except Exception as e:
            print(f"Error in handle_comparison_query: {e}")
            import traceback
            traceback.print_exc()
            
            # Return a more helpful error message
            return (f"I encountered an error while comparing the people you mentioned. "
                    f"Please try specifying their exact full names as they appear in the database. "
                    f"Error details: {str(e)}")
    def extract_person_skills(self, person_name: str, project_store: Dict) -> Dict[str, float]:
        """
        Extract skills for a person from their projects with confidence scores.
        
        Args:
            person_name: The person's name
            project_store: The project document store
            
        Returns:
            Dict[str, float]: Dictionary of skill -> confidence score
        """
        skills = {}
        person_projects = project_store['by_person'].get(person_name, [])
        
        for project in person_projects:
            content = project['content'].lower()
            project_role = project['project_role'].lower()
            
            # Extract skill mentions from content and role
            skill_patterns = [
                r'\b(aws|amazon|azure|microsoft|gcp|google cloud|lambda|s3|ec2|dynamodb|cosmos|cloud|kubernetes|k8s|docker|terraform|jenkins|gitlab|github|ci/cd|cicd|pipeline|deployment)\b',
                r'\b(python|java|javascript|typescript|node\.js|react|angular|vue|express|spring|django|flask|ruby|go|golang|rust|c\+\+|c#)\b',
                r'\b(devops|sre|cloud native|infrastructure|migration|architect|security|monitoring|machine learning|ml|ai|artificial intelligence|deep learning|nlp|neural network)\b',
                r'\b(sql|mysql|postgresql|mongodb|nosql|database|db|oracle|sqlite|redis|elasticsearch|kafka|spark|hadoop|web services|api|rest|graphql)\b',
                r'\b(android|ios|swift|kotlin|flutter|react native|mobile|unity|xamarin|cordova|ionic|pwa|progressive web app)\b',
                r'\b(html|css|sass|less|javascript|frontend|backend|fullstack|web|ui|ux|design|responsive|accessibility|seo)\b',
                r'\b(agile|scrum|kanban|waterfall|project management|jira|confluence|trello|asana|product owner|scrum master|sprint|lean|six sigma)\b',
                r'\b(testing|test automation|selenium|cypress|jest|mocha|chai|junit|testng|qunit|cucumber|bdd|tdd|ci|continuous integration)\b'
            ]
            
            # Combine content and role for skill extraction
            text = f"{content} {project_role}"
            
            for pattern in skill_patterns:
                matches = re.findall(pattern, text)
                for skill in matches:
                    # Calculate confidence score based on:
                    # - Frequency of mention
                    # - Presence in role title (higher confidence)
                    # - Presence in responsibilities section
                    frequency = text.count(skill)
                    in_role = skill in project_role
                    in_responsibilities = "responsibilities" in text and skill in text[text.index("responsibilities"):min(text.index("responsibilities") + 500, len(text))]
                    
                    confidence = min(1.0, 0.7 + (frequency * 0.05) + (0.2 if in_role else 0) + (0.1 if in_responsibilities else 0))
                    
                    if skill not in skills or confidence > skills[skill]:
                        skills[skill] = confidence
        
        return skills

    def extract_person_availability(self, person_name: str, project_store: Dict) -> Dict:
        """
        Extract availability information for a person based on their project history.
        
        Args:
            person_name: The person's name
            project_store: The project document store
            
        Returns:
            Dict: Dictionary with availability information
        """
        person_projects = project_store['by_person'].get(person_name, [])
        current_date = datetime.now()
        availability = {
            "current_project": None,
            "current_role": None,
            "project_end_date": None,
            "is_available": False,
            "availability_confidence": 0.0,
            "recent_projects": [],
            "average_project_duration_months": 0
        }
        
        if not person_projects:
            return availability
        
        # Parse project dates
        project_durations = []
        for project in person_projects:
            content = project['content']
            
            # Extract period information
            period_match = re.search(r"Period:\s*(\w+\s+\d{4})\s+to\s+(\w+\s+\d{4}|Present)", content)
            
            if period_match:
                start_date_str = period_match.group(1)
                end_date_str = period_match.group(2)
                
                try:
                    # Parse start date
                    start_date = datetime.strptime(start_date_str, "%B %Y")
                    
                    # Parse end date
                    if end_date_str.lower() == "present":
                        end_date = current_date
                        
                        # This is a current project
                        availability["current_project"] = project['project_name']
                        availability["current_role"] = project['project_role']
                        availability["project_end_date"] = "Present"
                        availability["is_available"] = False
                        availability["availability_confidence"] = 0.9
                    else:
                        end_date = datetime.strptime(end_date_str, "%B %Y")
                    
                    # Calculate duration in months
                    duration_months = ((end_date.year - start_date.year) * 12) + (end_date.month - start_date.month)
                    
                    # Add to project durations for average calculation
                    project_durations.append(duration_months)
                    
                    # Add to recent projects (last 2 years)
                    if (current_date - end_date).days <= 730:  # Roughly 2 years
                        availability["recent_projects"].append({
                            "name": project['project_name'],
                            "role": project['project_role'],
                            "duration_months": duration_months,
                            "end_date": end_date_str
                        })
                except ValueError:
                    # Skip if date parsing fails
                    continue
        
        # Calculate average project duration
        if project_durations:
            availability["average_project_duration_months"] = sum(project_durations) / len(project_durations)
        
        # Check if person is potentially available (no current project)
        if not availability["current_project"]:
            availability["is_available"] = True
            
            # Calculate confidence based on recency of last project
            if availability["recent_projects"]:
                most_recent = sorted(availability["recent_projects"], key=lambda x: x["end_date"], reverse=True)[0]
                
                # If last project ended very recently, lower confidence in availability
                if "Present" in most_recent["end_date"]:
                    availability["is_available"] = False
                    availability["availability_confidence"] = 0.9
                else:
                    try:
                        last_end_date = datetime.strptime(most_recent["end_date"], "%B %Y")
                        months_since_last_project = ((current_date.year - last_end_date.year) * 12) + (current_date.month - last_end_date.month)
                        
                        # Higher confidence if more time has passed since last project
                        if months_since_last_project <= 1:
                            availability["availability_confidence"] = 0.7
                        elif months_since_last_project <= 3:
                            availability["availability_confidence"] = 0.8
                        else:
                            availability["availability_confidence"] = 0.9
                    except ValueError:
                        availability["availability_confidence"] = 0.6  # Default if parsing fails
            else:
                availability["availability_confidence"] = 0.5  # Lower confidence if no recent projects
        
        return availability

    def compare_people(self, people_names: List[str], query: str, project_store: Dict, query_embedding) -> Dict:
        """
        Compare multiple people based on their skills, projects, and availability.
        
        Args:
            people_names: List of people to compare
            query: The original query
            project_store: The project document store
            query_embedding: Embedding of the query
            
        Returns:
            Dict: Structured comparison results
        """
        comparison_results = {
            "people": [],
            "common_skills": [],
            "skill_comparison": {},
            "project_comparison": {},
            "availability_comparison": {},
            "focus_areas": {
                "skills": False,
                "projects": False,
                "availability": False
            }
        }
        
        # Determine focus areas from query
        query_lower = query.lower()
        if any(term in query_lower for term in ["skill", "know", "experience", "expert", "proficient", "familiar"]):
            comparison_results["focus_areas"]["skills"] = True
        
        if any(term in query_lower for term in ["project", "work", "built", "developed", "led", "managed"]):
            comparison_results["focus_areas"]["projects"] = True
        
        if any(term in query_lower for term in ["available", "availability", "schedule", "free", "current", "busy"]):
            comparison_results["focus_areas"]["availability"] = True
        
        # If no specific focus area, include all
        if not any(comparison_results["focus_areas"].values()):
            comparison_results["focus_areas"]["skills"] = True
            comparison_results["focus_areas"]["projects"] = True
            comparison_results["focus_areas"]["availability"] = True
        
        # Extract and compare skills
        all_skills = {}
        person_skills = {}
        
        for person_name in people_names:
            # Get person's skills
            skills = self.extract_person_skills(person_name, project_store)
            person_skills[person_name] = skills
            
            # Update all skills
            for skill, confidence in skills.items():
                if skill not in all_skills:
                    all_skills[skill] = 0
                all_skills[skill] += 1
        
        # Find common skills (shared by all people)
        comparison_results["common_skills"] = [
            skill for skill, count in all_skills.items() 
            if count == len(people_names) and all(person_skills[person].get(skill, 0) > 0.7 for person in people_names)
        ]
        
        # Create skill comparison
        for skill in all_skills.keys():
            comparison_results["skill_comparison"][skill] = {
                person_name: person_skills[person_name].get(skill, 0) 
                for person_name in people_names
            }
        
        # Extract and compare projects
        for person_name in people_names:
            person_projects = project_store['by_person'].get(person_name, [])
            
            # Create person entry
            person_data = {
                "name": person_name,
                "project_count": len(person_projects),
                "skills": [skill for skill, confidence in sorted(person_skills[person_name].items(), key=lambda x: x[1], reverse=True) if confidence > 0.7],
                "top_projects": [],
                "availability": self.extract_person_availability(person_name, project_store)
            }
            
            # Add top projects
            for project in person_projects[:3]:  # Top 3 projects
                project_data = {
                    "name": project['project_name'],
                    "role": project['project_role'],
                    "content": project['content']
                }
                
                # Extract period if available
                period_match = re.search(r"Period:\s*(.+?)\s+to\s+(.+?)(?:\n|$)", project['content'])
                if period_match:
                    project_data["period"] = f"{period_match.group(1)} to {period_match.group(2)}"
                
                # Extract responsibilities
                resp_match = re.search(r"Responsibilities?:\s*(.*?)(?=\n\n|$)", project['content'], re.DOTALL)
                if resp_match:
                    project_data["responsibilities"] = resp_match.group(1).strip()
                
                person_data["top_projects"].append(project_data)
            
            comparison_results["people"].append(person_data)
        
        # Compare project experience
        comparison_results["project_comparison"] = {
            "project_count": {person["name"]: person["project_count"] for person in comparison_results["people"]},
            "common_roles": self.identify_common_roles(people_names, project_store),
            "unique_projects": self.identify_unique_projects(people_names, project_store)
        }
        
        # Compare availability
        comparison_results["availability_comparison"] = {
            person["name"]: person["availability"] for person in comparison_results["people"]
        }
        
        return comparison_results

    def identify_common_roles(self, people_names: List[str], project_store: Dict) -> List[str]:
        """
        Identify common roles across the people being compared.
        
        Args:
            people_names: List of people to compare
            project_store: The project document store
            
        Returns:
            List[str]: List of common roles
        """
        person_roles = {}
        
        for person_name in people_names:
            person_projects = project_store['by_person'].get(person_name, [])
            person_roles[person_name] = set()
            
            for project in person_projects:
                # Normalize role for better matching
                role = project['project_role'].lower()
                # Extract the core role, ignoring specifics
                core_role = re.sub(r'\b(senior|junior|lead|principal|staff|associate)\b', '', role).strip()
                person_roles[person_name].add(core_role)
        
        # Find roles that all people have in common
        if not people_names:
            return []
            
        common_roles = person_roles[people_names[0]].copy()
        for person_name in people_names[1:]:
            common_roles &= person_roles[person_name]
        
        return list(common_roles)

    def identify_unique_projects(self, people_names: List[str], project_store: Dict) -> Dict[str, List[str]]:
        """
        Identify unique projects for each person that others haven't worked on.
        
        Args:
            people_names: List of people to compare
            project_store: The project document store
            
        Returns:
            Dict[str, List[str]]: Dictionary of person -> list of unique projects
        """
        all_project_names = set()
        person_project_names = {}
        
        for person_name in people_names:
            person_projects = project_store['by_person'].get(person_name, [])
            project_names = {project['project_name'] for project in person_projects}
            
            person_project_names[person_name] = project_names
            all_project_names.update(project_names)
        
        # Find unique projects for each person
        unique_projects = {}
        for person_name in people_names:
            other_people_projects = set()
            for other_name in people_names:
                if other_name != person_name:
                    other_people_projects.update(person_project_names[other_name])
            
            unique_projects[person_name] = list(person_project_names[person_name] - other_people_projects)
        
        return unique_projects
    
    def analyze_query_intent(self, query: str, query_embedding) -> Dict:
        """
        Analyze query intent to determine skills and focus areas without relying on predefined keywords.
        Uses semantic analysis and more flexible pattern matching.
        
        Args:
            query: The user query
            query_embedding: Embedding vector of the query
            
        Returns:
            Dict: Dictionary with query intent analysis
        """
        query_lower = query.lower()
        
        # Initialize intent analysis
        intent = {
            "skills_focus": [],       # Skills mentioned or implied in query
            "project_types": [],      # Types of projects mentioned
            "comparison_criteria": [] # What criteria to use for comparison
        }
        
        # Step 1: Use regex to extract potential skill and technology terms
        # This looks for noun phrases that might be skills/technologies
        skill_candidates = set()
        
        # Look for technical terms and noun phrases that might be skills
        patterns = [
            # Technology with version numbers
            r'\b([a-zA-Z]+(?:\s[a-zA-Z]+){0,2})\s+\d+(?:\.\d+){0,2}\b',
            # Capitalized terms that might be technologies or frameworks
            r'\b([A-Z][a-zA-Z0-9]*(?:\.[A-Z][a-zA-Z0-9]*)*)\b',
            # Terms followed by common technical qualifiers
            r'\b([a-zA-Z]+(?:\s[a-zA-Z]+){0,2})\s+(?:framework|language|platform|database|stack|service|api|sdk)\b',
            # Terms preceded by technical adjectives
            r'\b(?:programming|scripting|markup|query|cloud|containerization|orchestration|virtualization)\s+([a-zA-Z]+(?:\s[a-zA-Z]+){0,2})\b',
            # Noun phrases that might be technologies
            r'\b([a-zA-Z]+(?:\s[a-zA-Z]+){0,2})\s+(?:development|programming|engineering|architecture|deployment|testing)\b',
            # General noun phrases that might be relevant
            r'\b([a-zA-Z]+(?:[-_][a-zA-Z]+)*)\b'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, query_lower)
            for match in matches:
                if isinstance(match, tuple):
                    for group in match:
                        if len(group) > 2:  # Ignore very short matches
                            skill_candidates.add(group.strip())
                elif len(match) > 2:  # Ignore very short matches
                    skill_candidates.add(match.strip())
        
        # Step 2: Filter candidates to remove common words and non-technical terms
        stop_words = {
            'and', 'or', 'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'but', 'if', 'then', 'else', 'when',
            'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most',
            'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
            'than', 'too', 'very', 'can', 'will', 'just', 'should', 'now', 'between',
            'compare', 'vs', 'versus', 'about', 'person', 'people', 'who', 'better', 'best',
            'good', 'great', 'excellent', 'top', 'know', 'project', 'projects', 'skill', 'skills'
        }
        
        # Remove stop words and common non-technical terms
        skill_candidates = {term for term in skill_candidates if term not in stop_words}
        
        # Step 3: Check for project type indicators
        project_type_indicators = [
            (r'\b(web|mobile|desktop|cloud|api|database|frontend|backend|fullstack)\b', lambda m: m.group(1)),
            (r'\b(app|application|service|platform|system|pipeline)\b', lambda m: m.group(1)),
            (r'\b(development|migration|integration|automation)\b', lambda m: m.group(1)),
            (r'\b([a-zA-Z]+)\s+(?:project|application|system)\b', lambda m: m.group(1))
        ]
        
        for pattern, extractor in project_type_indicators:
            for match in re.finditer(pattern, query_lower):
                term = extractor(match)
                if term:
                    intent["project_types"].append(term)
        
        # Step 4: Use vector similarity to evaluate candidates
        # Create embeddings for some common tech categories to use for classification
        tech_categories = {
            "programming_languages": "Python Java JavaScript TypeScript C# C++ Ruby Go Swift Kotlin",
            "web_frameworks": "React Angular Vue Django Flask Spring Express ASP.NET Ruby on Rails",
            "databases": "SQL MySQL PostgreSQL MongoDB Oracle NoSQL Redis Cassandra DynamoDB",
            "cloud_platforms": "AWS Amazon Web Services Azure GCP Google Cloud IBM Cloud Oracle Cloud",
            "devops_tools": "Docker Kubernetes Jenkins GitHub Actions GitLab CI/CD Travis CI CircleCI",
            "infrastructure": "Terraform CloudFormation Ansible Puppet Chef Salt Vagrant",
            "big_data": "Hadoop Spark Kafka Flink Storm Hive Pig Presto",
            "ai_ml": "Machine Learning TensorFlow PyTorch scikit-learn NLTK Computer Vision"
        }
        
        tech_category_embeddings = {}
        for category, terms in tech_categories.items():
            try:
                tech_category_embeddings[category] = self.cached_embedding(terms)
            except Exception as e:
                print(f"Error creating embedding for {category}: {e}")
        
        # Evaluate each candidate
        evaluated_candidates = []
        for candidate in skill_candidates:
            try:
                # Skip very common words or short terms unlikely to be skills
                if len(candidate) < 3 or candidate in stop_words:
                    continue
                    
                candidate_embedding = self.cached_embedding(candidate)
                
                # Calculate similarity to tech categories
                highest_similarity = 0
                best_category = None
                
                for category, category_embedding in tech_category_embeddings.items():
                    similarity = self.cosine_similarity_calc([candidate_embedding], [category_embedding])
                    similarity_value = float(similarity) if not hasattr(similarity, 'shape') else float(similarity[0][0])
                    
                    if similarity_value > highest_similarity:
                        highest_similarity = similarity_value
                        best_category = category
                
                # Also calculate similarity to query itself
                query_similarity = self.cosine_similarity_calc([candidate_embedding], [query_embedding])
                query_sim_value = float(query_similarity) if not hasattr(query_similarity, 'shape') else float(query_similarity[0][0])
                
                # Evaluate if this is likely a technical term
                if highest_similarity > 0.4 or query_sim_value > 0.6:
                    evaluated_candidates.append({
                        "term": candidate,
                        "category": best_category,
                        "category_similarity": highest_similarity,
                        "query_similarity": query_sim_value,
                        "combined_score": (highest_similarity * 0.7) + (query_sim_value * 0.3)
                    })
                    
            except Exception as e:
                print(f"Error evaluating candidate '{candidate}': {e}")
        
        # Sort candidates by combined score
        evaluated_candidates.sort(key=lambda x: x["combined_score"], reverse=True)
        
        # Take top candidates as skills focus (up to 5)
        for candidate in evaluated_candidates[:5]:
            if candidate["combined_score"] > 0.4:  # Threshold to ensure relevance
                intent["skills_focus"].append(candidate["term"])
        
        # Step 5: Determine comparison criteria from query semantics
        comparison_indicators = [
            (r'\b(skills?|experience|know\w*|familiar|proficient|expert)\b', "skill_proficiency"),
            (r'\b(project|work\w*|built|develop\w*|implement\w*)\b', "project_experience"),
            (r'\b(available|busy|schedule|timeline|when|time)\b', "availability"),
            (r'\b(role|position|title|responsibility)\b', "roles"),
            (r'\b(tool|technolog\w*|platform|framework|stack)\b', "technologies"),
            (r'\b(better|best|compare|vs|versus|against|rank\w*|rate|prefer\w*)\b', "ranking")
        ]
        
        for pattern, criterion in comparison_indicators:
            if re.search(pattern, query_lower):
                if criterion not in intent["comparison_criteria"]:
                    intent["comparison_criteria"].append(criterion)
        
        # If no specific criteria found, add defaults
        if not intent["comparison_criteria"]:
            intent["comparison_criteria"] = ["skill_proficiency", "project_experience", "technologies"]
        
        # Always include ranking as we'll provide ranking at the end
        if "ranking" not in intent["comparison_criteria"]:
            intent["comparison_criteria"].append("ranking")
        
        return intent

    def rank_candidates(self, comparison_results: Dict, query_intent: Dict) -> List[Dict]:
        """
        Rank candidates based on skills focus and comparison criteria.
        
        Args:
            comparison_results: The comparison results
            query_intent: The analyzed query intent
            
        Returns:
            List[Dict]: Ranked candidates with scores and reasons
        """
        people = comparison_results["people"]
        skills_focus = query_intent["skills_focus"]
        comparison_criteria = query_intent["comparison_criteria"]
        project_types = query_intent["project_types"]
        
        rankings = []
        
        for person in people:
            person_name = person["name"]
            scores = {
                "skill_proficiency": 0,
                "project_experience": 0,
                "availability": 0,
                "roles": 0,
                "technologies": 0,
                "overall": 0
            }
            
            reasons = []
            
            # Score skill proficiency
            if "skill_proficiency" in comparison_criteria:
                skill_scores = []
                matched_skills = []
                
                # Score based on skills mentioned in query
                if skills_focus:
                    for focus_skill in skills_focus:
                        best_match_score = 0
                        best_match_skill = None
                        
                        for skill, confidence in comparison_results["skill_comparison"].items():
                            if person_name in confidence:
                                # Check if this skill matches focus skill
                                if focus_skill.lower() in skill.lower() or skill.lower() in focus_skill.lower():
                                    score = confidence[person_name]
                                    if score > best_match_score:
                                        best_match_score = score
                                        best_match_skill = skill
                        
                        if best_match_skill:
                            skill_scores.append(best_match_score)
                            matched_skills.append(best_match_skill)
                
                # If no query skills matched, score based on overall skill level
                if not skill_scores:
                    for skill, confidence in comparison_results["skill_comparison"].items():
                        if person_name in confidence and confidence[person_name] > 0.7:
                            skill_scores.append(confidence[person_name])
                            if len(matched_skills) < 3:  # Limit to top 3 skills
                                matched_skills.append(skill)
                
                # Calculate skill score
                if skill_scores:
                    scores["skill_proficiency"] = sum(skill_scores) / len(skill_scores)
                    
                    # Add reason
                    if matched_skills:
                        proficiency_levels = []
                        for skill in matched_skills:
                            confidence = comparison_results["skill_comparison"].get(skill, {}).get(person_name, 0)
                            if confidence > 0.9:
                                proficiency_levels.append(f"Expert in {skill}")
                            elif confidence > 0.8:
                                proficiency_levels.append(f"Advanced in {skill}")
                            elif confidence > 0.7:
                                proficiency_levels.append(f"Proficient in {skill}")
                            elif confidence > 0.6:
                                proficiency_levels.append(f"Familiar with {skill}")
                        
                        reasons.append(", ".join(proficiency_levels[:3]))
            
            # Score project experience
            if "project_experience" in comparison_criteria:
                project_count = comparison_results["project_comparison"]["project_count"].get(person_name, 0)
                
                # Normalize project count (assuming max 10 projects is excellent)
                normalized_project_count = min(project_count / 10, 1.0)
                
                # Score based on project types if specified
                project_type_score = 0
                relevant_projects = []
                
                if project_types:
                    for p in comparison_results["people"]:
                        if p["name"] == person_name:
                            for project in p.get("top_projects", []):
                                project_name = project.get("name", "")
                                project_content = project.get("content", "")
                                
                                # Check if this project matches any of the project types
                                for project_type in project_types:
                                    if project_type.lower() in project_name.lower() or project_type.lower() in project_content.lower():
                                        project_type_score += 0.2  # Boost score for each matching project
                                        relevant_projects.append(project_name)
                                        break
                
                # Combine project scores
                scores["project_experience"] = (normalized_project_count * 0.6) + (min(project_type_score, 0.4))
                
                # Add reason
                if project_count > 0:
                    reason = f"{project_count} projects"
                    if relevant_projects:
                        reason += f" including relevant projects: {', '.join(relevant_projects[:2])}"
                    reasons.append(reason)
            
            # Score availability
            if "availability" in comparison_criteria:
                avail = comparison_results["availability_comparison"].get(person_name, {})
                
                # Higher score if available
                if avail.get("is_available", False):
                    scores["availability"] = 0.9
                    reasons.append("Currently available")
                else:
                    # Score based on how soon they'll be available
                    current_project = avail.get("current_project")
                    if current_project:
                        end_date = avail.get("project_end_date")
                        if end_date and end_date.lower() != "present":
                            # Calculate approximate time until available (very simplified)
                            scores["availability"] = 0.3
                            reasons.append(f"Working on {current_project} until {end_date}")
                        else:
                            scores["availability"] = 0.1
                            reasons.append(f"Working on ongoing project: {current_project}")
                    else:
                        scores["availability"] = 0.5
                        reasons.append("No current project information")
            
            # Score roles
            if "roles" in comparison_criteria:
                relevant_roles = 0
                person_roles = []
                
                for p in comparison_results["people"]:
                    if p["name"] == person_name:
                        for project in p.get("top_projects", []):
                            role = project.get("role", "").strip()
                            if role and role not in person_roles:
                                person_roles.append(role)
                                
                                # Check if role is relevant to skills focus or project types
                                if any(focus.lower() in role.lower() for focus in skills_focus):
                                    relevant_roles += 1
                
                # Score based on diversity and relevance of roles
                role_diversity = min(len(person_roles) / 5, 1.0)  # Normalize to max 5 roles
                role_relevance = min(relevant_roles / 3, 1.0)     # Normalize to max 3 relevant roles
                
                scores["roles"] = (role_diversity * 0.6) + (role_relevance * 0.4)
                
                # Add reason
                if person_roles:
                    reasons.append(f"Roles: {', '.join(person_roles[:3])}")
            
            # Score technologies
            if "technologies" in comparison_criteria:
                tech_count = 0
                relevant_tech = 0
                person_techs = []
                
                # From skills
                for skill in comparison_results["skill_comparison"]:
                    if person_name in comparison_results["skill_comparison"][skill] and comparison_results["skill_comparison"][skill][person_name] > 0.6:
                        tech_count += 1
                        if skill not in person_techs:
                            person_techs.append(skill)
                        
                        # Check if relevant to skills focus
                        if any(focus.lower() in skill.lower() or skill.lower() in focus.lower() for focus in skills_focus):
                            relevant_tech += 1
                
                # Normalize technology scores
                tech_diversity = min(tech_count / 10, 1.0)        # Normalize to max 10 techs
                tech_relevance = min(relevant_tech / 5, 1.0)      # Normalize to max 5 relevant techs
                
                scores["technologies"] = (tech_diversity * 0.5) + (tech_relevance * 0.5)
                
                # Add reason
                if person_techs:
                    # Prioritize techs related to skills focus
                    if skills_focus:
                        relevant_techs = [tech for tech in person_techs if any(focus.lower() in tech.lower() or tech.lower() in focus.lower() for focus in skills_focus)]
                        top_techs = relevant_techs[:3] if relevant_techs else person_techs[:3]
                    else:
                        top_techs = person_techs[:3]
                        
                    reasons.append(f"Technologies: {', '.join(top_techs)}")
            
            # Calculate overall score with weighting based on criteria
            weights = {
                "skill_proficiency": 0.3,
                "project_experience": 0.25,
                "availability": 0.15,
                "roles": 0.15,
                "technologies": 0.15
            }
            
            # Adjust weights based on which criteria are included
            total_weight = sum(weights[criterion] for criterion in comparison_criteria if criterion != "ranking")
            if total_weight > 0:
                normalized_weights = {criterion: weights[criterion] / total_weight for criterion in weights}
                
                # Calculate weighted average
                scores["overall"] = sum(scores[criterion] * normalized_weights[criterion] 
                                    for criterion in comparison_criteria if criterion != "ranking")
            
            rankings.append({
                "name": person_name,
                "scores": scores,
                "reasons": reasons
            })
        
        # Sort by overall score
        rankings.sort(key=lambda x: x["scores"]["overall"], reverse=True)
        return rankings
    def format_comparison_results_tabular(self, comparison_results: Dict, query: str) -> str:
        """
        Format comparison results in a tabular format using markdown.
        Optimized for comparing up to 4 people, with intelligent analysis and ranking.
        
        Args:
            comparison_results: The comparison results
            query: The original user query
            
        Returns:
            str: Formatted tabular comparison output
        """
        people_names = [person["name"] for person in comparison_results["people"]]
        
        # Limit to 4 people maximum
        if len(people_names) > 4:
            people_names = people_names[:4]
            # Filter comparison results to include only these people
            for key in ["skill_comparison", "project_comparison", "availability_comparison"]:
                if key in comparison_results:
                    if isinstance(comparison_results[key], dict):
                        for subkey in list(comparison_results[key].keys()):
                            if isinstance(comparison_results[key][subkey], dict):
                                comparison_results[key][subkey] = {
                                    k: v for k, v in comparison_results[key][subkey].items() 
                                    if k in people_names
                                }
        
        # Get query embedding for analysis
        query_embedding = self.cached_embedding(query)
        
        # Analyze query intent
        query_intent = self.analyze_query_intent(query, query_embedding)
        
        formatted_output = []
        
        # Generate title
        if len(people_names) == 2:
            formatted_output.append(f"# Comparison: {people_names[0]} vs. {people_names[1]}")
        else:
            formatted_output.append(f"# Comparison of {len(people_names)} People")
            formatted_output.append(people_names[0] + " vs. " + " vs. ".join(people_names[1:]))
        
        # If query indicates skills focus, add it to the title
        skills_focus = query_intent["skills_focus"]
        if skills_focus:
            formatted_output.append(f"\nFocus: {', '.join(skills_focus)}")
        
        # Add detected focus areas
        project_types = query_intent["project_types"]
        if project_types:
            formatted_output.append(f"Project Types: {', '.join(project_types)}")
        
        # Add query-focused skills first if mentioned in query
        if skills_focus and comparison_results["focus_areas"]["skills"]:
            formatted_output.append("\n## Query-Focused Skills\n")
            
            # Create skill comparison table header
            skill_table = ["| Skill | " + " | ".join(people_names) + " |",
                        "| --- | " + " | ".join(["---"] * len(people_names)) + " |"]
            
            # For each skill focus, find matching skills in the comparison results
            for focus_skill in skills_focus:
                matching_skills = []
                
                # Find exact and partial matches
                for skill in comparison_results["skill_comparison"].keys():
                    # Check for matches in both directions
                    if focus_skill.lower() in skill.lower() or skill.lower() in focus_skill.lower():
                        matching_skills.append(skill)
                
                # If no exact matches, try fuzzy matching
                if not matching_skills:
                    for skill in comparison_results["skill_comparison"].keys():
                        # Simple fuzzy matching using character overlap
                        focus_chars = set(focus_skill.lower())
                        skill_chars = set(skill.lower())
                        overlap = len(focus_chars & skill_chars) / len(focus_chars | skill_chars)
                        
                        if overlap > 0.6:  # 60% character overlap threshold
                            matching_skills.append(skill)
                
                # Add rows for matching skills
                for skill in matching_skills:
                    row = [f"**{skill}**"]  # Bold to highlight query-focused skills
                    for person in people_names:
                        confidence = comparison_results["skill_comparison"][skill].get(person, 0)
                        if confidence > 0.9:
                            row.append("Expert")
                        elif confidence > 0.8:
                            row.append("Advanced")
                        elif confidence > 0.7:
                            row.append("Proficient")
                        elif confidence > 0.6:
                            row.append("Familiar")
                        else:
                            row.append("-")
                    
                    skill_table.append("| " + " | ".join(row) + " |")
            
            # Only add the table if we found matching skills
            if len(skill_table) > 2:
                formatted_output.extend(skill_table)
            else:
                alternative_skills = []
                
                # Look for alternative skills that might be relevant
                for focus_skill in skills_focus:
                    focus_embedding = self.cached_embedding(focus_skill)
                    
                    for skill in list(comparison_results["skill_comparison"].keys())[:20]:  # Limit to first 20 for performance
                        try:
                            skill_embedding = self.cached_embedding(skill)
                            similarity = self.cosine_similarity_calc([focus_embedding], [skill_embedding])
                            sim_value = float(similarity) if not hasattr(similarity, 'shape') else float(similarity[0][0])
                            
                            if sim_value > 0.7:  # Semantic similarity threshold
                                alternative_skills.append((skill, sim_value))
                        except Exception as e:
                            print(f"Error comparing skill embeddings: {e}")
                
                # Sort by similarity and take top 3
                alternative_skills.sort(key=lambda x: x[1], reverse=True)
                
                if alternative_skills:
                    formatted_output.append(f"No exact matches found for '{', '.join(skills_focus)}', but found these related skills:")
                    
                    # Create table for alternative skills
                    alt_skill_table = ["| Skill | " + " | ".join(people_names) + " |",
                                    "| --- | " + " | ".join(["---"] * len(people_names)) + " |"]
                    
                    for skill, _ in alternative_skills[:3]:
                        row = [skill]
                        for person in people_names:
                            confidence = comparison_results["skill_comparison"][skill].get(person, 0)
                            if confidence > 0.9:
                                row.append("Expert")
                            elif confidence > 0.8:
                                row.append("Advanced")
                            elif confidence > 0.7:
                                row.append("Proficient")
                            elif confidence > 0.6:
                                row.append("Familiar")
                            else:
                                row.append("-")
                        
                        alt_skill_table.append("| " + " | ".join(row) + " |")
                    
                    formatted_output.extend(alt_skill_table)
                else:
                    formatted_output.append(f"No skills found matching '{', '.join(skills_focus)}'.")
        
        # Add general skills comparison
        if comparison_results["focus_areas"]["skills"]:
            formatted_output.append("\n## Skills Comparison\n")
            
            # Create skill comparison table header
            skill_table = ["| Skill | " + " | ".join(people_names) + " |",
                        "| --- | " + " | ".join(["---"] * len(people_names)) + " |"]
            
            # Get top skills (with confidence > 0.7 for any person)
            relevant_skills = [
                skill for skill, person_confidences in comparison_results["skill_comparison"].items()
                if any(confidence > 0.7 for confidence in person_confidences.values())
            ]
            
            # Sort skills by average confidence
            sorted_skills = sorted(
                relevant_skills,
                key=lambda skill: sum(comparison_results["skill_comparison"][skill].values()) / len(people_names),
                reverse=True
            )
            
            # Filter out skills already shown in query-focused section
            if skills_focus:
                filtered_skills = []
                for skill in sorted_skills:
                    already_shown = False
                    for focus_skill in skills_focus:
                        if focus_skill.lower() in skill.lower() or skill.lower() in focus_skill.lower():
                            already_shown = True
                            break
                    if not already_shown:
                        filtered_skills.append(skill)
                sorted_skills = filtered_skills
            
            # Create rows for skills
            for skill in sorted_skills[:15]:  # Limit to top 15 skills
                row = [skill]
                for person in people_names:
                    confidence = comparison_results["skill_comparison"][skill].get(person, 0)
                    if confidence > 0.9:
                        row.append("Expert")
                    elif confidence > 0.8:
                        row.append("Advanced")
                    elif confidence > 0.7:
                        row.append("Proficient")
                    elif confidence > 0.6:
                        row.append("Familiar")
                    else:
                        row.append("-")
                
                skill_table.append("| " + " | ".join(row) + " |")
            
            # Add skill table if we have skills to show
            if len(skill_table) > 2:
                formatted_output.extend(skill_table)
        
        # Add project comparison table if focus area includes projects
        if comparison_results["focus_areas"]["projects"]:
            formatted_output.append("\n## Project Experience\n")
            
            # Create project metrics table
            project_table = ["| Metric | " + " | ".join(people_names) + " |",
                        "| --- | " + " | ".join(["---"] * len(people_names)) + " |"]
            
            # Add project count row
            project_counts = []
            for person in people_names:
                count = comparison_results["project_comparison"]["project_count"].get(person, 0)
                project_counts.append(str(count))
            project_table.append("| Total Projects | " + " | ".join(project_counts) + " |")
            
            # Add query-specific project counts if project types are mentioned
            if project_types:
                for project_type in project_types:
                    type_counts = []
                    for person in people_names:
                        # Count projects of this type for each person
                        count = 0
                        for p in comparison_results["people"]:
                            if p["name"] == person:
                                for project in p.get("top_projects", []):
                                    content = (project.get("content", "") + " " + project.get("name", "")).lower()
                                    if project_type.lower() in content:
                                        count += 1
                        type_counts.append(str(count))
                    
                    project_table.append(f"| {project_type.title()} Projects | " + " | ".join(type_counts) + " |")
            
            # Extract and add average project duration
            durations = []
            for person in people_names:
                avail = comparison_results["availability_comparison"].get(person, {})
                duration = avail.get("average_project_duration_months", 0)
                durations.append(f"{duration:.1f} months")
            project_table.append("| Avg Project Duration | " + " | ".join(durations) + " |")
            
            # Add project table
            formatted_output.extend(project_table)
            
            # Add project roles table
            formatted_output.append("\n## Project Roles\n")
            
            # Create roles table
            roles_table = ["| Role | " + " | ".join(people_names) + " |",
                        "| --- | " + " | ".join(["---"] * len(people_names)) + " |"]
            
            # Get all unique roles across all people
            all_roles = set()
            query_relevant_roles = set()
            
            for person in comparison_results["people"]:
                if person["name"] not in people_names:
                    continue
                    
                person_projects = person.get("top_projects", [])
                for project in person_projects:
                    role = project.get("role", "").strip()
                    if not role:
                        continue
                        
                    # Normalize role names for better comparison
                    role = re.sub(r'\b(junior|senior|lead|principal|staff)\b', '', role, flags=re.IGNORECASE).strip()
                    all_roles.add(role)
                    
                    # Check if this role is relevant to query focus
                    if skills_focus and any(focus.lower() in role.lower() for focus in skills_focus):
                        query_relevant_roles.add(role)
                    if project_types and any(proj_type.lower() in role.lower() for proj_type in project_types):
                        query_relevant_roles.add(role)
            
            # First add query-relevant roles
            for role in sorted(query_relevant_roles):
                row = [f"**{role}**"]  # Bold to highlight query-relevant roles
                for person in people_names:
                    # Check if this person has this role in any project
                    has_role = False
                    for p in comparison_results["people"]:
                        if p["name"] == person:
                            for project in p.get("top_projects", []):
                                project_role = project.get("role", "").strip()
                                normalized_role = re.sub(r'\b(junior|senior|lead|principal|staff)\b', '', project_role, flags=re.IGNORECASE).strip()
                                if normalized_role == role:
                                    has_role = True
                                    break
                    
                    row.append("✓" if has_role else "✗")
                
                roles_table.append("| " + " | ".join(row) + " |")
            
            # Then add other roles
            for role in sorted(all_roles - query_relevant_roles):
                row = [role]
                for person in people_names:
                    # Check if this person has this role in any project
                    has_role = False
                    for p in comparison_results["people"]:
                        if p["name"] == person:
                            for project in p.get("top_projects", []):
                                project_role = project.get("role", "").strip()
                                normalized_role = re.sub(r'\b(junior|senior|lead|principal|staff)\b', '', project_role, flags=re.IGNORECASE).strip()
                                if normalized_role == role:
                                    has_role = True
                                    break
                    
                    row.append("✓" if has_role else "✗")
                
                roles_table.append("| " + " | ".join(row) + " |")
            
            # Add roles table
            formatted_output.extend(roles_table)
            
            # Add unique projects table
            unique_projects = comparison_results["project_comparison"]["unique_projects"]
            if any(projects for person, projects in unique_projects.items() if person in people_names):
                formatted_output.append("\n## Unique Projects\n")
                
                unique_table = ["| Person | Unique Projects |",
                            "| --- | --- |"]
                
                for person in people_names:
                    projects = unique_projects.get(person, [])
                    
                    # Prioritize projects matching query focus
                    if projects and (skills_focus or project_types):
                        # Sort projects to show query-relevant ones first
                        query_relevant = []
                        other_projects = []
                        
                        focus_terms = skills_focus + project_types
                        
                        for proj in projects[:10]:  # Limit to first 10 to avoid excessive processing
                            if any(term.lower() in proj.lower() for term in focus_terms):
                                query_relevant.append(f"**{proj}**")  # Bold to highlight
                            else:
                                other_projects.append(proj)
                        
                        # Combine, showing relevant ones first
                        sorted_projects = query_relevant + other_projects
                        
                        if sorted_projects:
                            unique_table.append(f"| {person} | {', '.join(sorted_projects[:5])}" + (", ..." if len(sorted_projects) > 5 else "") + " |")
                        else:
                            unique_table.append(f"| {person} | None |")
                    elif projects:
                        unique_table.append(f"| {person} | {', '.join(projects[:5])}" + (", ..." if len(projects) > 5 else "") + " |")
                    else:
                        unique_table.append(f"| {person} | None |")
                
                formatted_output.extend(unique_table)
        
        # Add technologies comparison
        formatted_output.append("\n## Technologies & Tools\n")
        
        # Gather technologies from skills and projects
        all_technologies = set()
        person_technologies = {person: set() for person in people_names}
        
        # Standard technology keywords to look for - expanded to catch more technologies
        technology_keywords = [
            "aws", "azure", "gcp", "cloud", "kubernetes", "docker", "terraform", 
            "jenkins", "gitlab", "github", "python", "java", "javascript", "c#",
            "react", "angular", "node", "spring", "django", "flask", "sql", "nosql",
            "mongodb", "postgresql", "mysql", "oracle", "redis", "kafka", "hadoop",
            "spark", "elasticsearch", "linux", "windows", "ios", "android", "mobile",
            # Add more technologies
            "typescript", "go", "ruby", "php", "rust", "swift", "kotlin", "objective-c",
            "vue", "express", "asp.net", "laravel", "symfony", "rails", "jquery",
            "dynamodb", "cassandra", "couchdb", "firebase", "neo4j", "graphql",
            "docker-compose", "swarm", "istio", "helm", "openshift", "lambda",
            "ec2", "s3", "ecs", "fargate", "eks", "rds", "aurora", "redshift",
            "cloudformation", "pulumi", "ansible", "puppet", "chef", "datadog",
            "prometheus", "grafana", "elk", "oauth", "jwt", "rest", "soap", "grpc"
        ]
        
        for person in people_names:
            # From skills
            for skill in comparison_results["skill_comparison"]:
                if person in comparison_results["skill_comparison"][skill] and comparison_results["skill_comparison"][skill][person] > 0.6:
                    skill_lower = skill.lower()
                    if any(tech in skill_lower for tech in technology_keywords):
                        person_technologies[person].add(skill_lower)
                        all_technologies.add(skill_lower)
            
            # From projects
            for p in comparison_results["people"]:
                if p["name"] == person:
                    for project in p.get("top_projects", []):
                        content = (project.get("content", "") + " " + project.get("role", "") + " " + project.get("name", "")).lower()
                        
                        for tech in technology_keywords:
                            if tech in content:
                                person_technologies[person].add(tech)
                                all_technologies.add(tech)
        
        # Add query-focused technologies first if query mentions technologies
        query_techs = set()
        if skills_focus:
            for focus in skills_focus:
                for tech in all_technologies:
                    if focus.lower() in tech or tech in focus.lower():
                        query_techs.add(tech)
        
        if query_techs:
            formatted_output.append("### Query-Focused Technologies\n")
            
            # Create technologies table header
            tech_table = ["| Technology | " + " | ".join(people_names) + " |",
                        "| --- | " + " | ".join(["---"] * len(people_names)) + " |"]
            
            # Add rows for query-related technologies
            for tech in sorted(query_techs):
                row = [tech.title()]  # Capitalize technology name
                for person in people_names:
                    has_tech = tech in person_technologies[person]
                    row.append("✓" if has_tech else "✗")
                
                tech_table.append("| " + " | ".join(row) + " |")
            
            formatted_output.extend(tech_table)
            
            # Remove query technologies from all technologies to avoid duplication
            all_technologies -= query_techs
        
        # Create general technologies table
        tech_table = ["| Technology | " + " | ".join(people_names) + " |",
                    "| --- | " + " | ".join(["---"] * len(people_names)) + " |"]
        
        # Sort technologies alphabetically
        for tech in sorted(all_technologies):
            row = [tech.title()]  # Capitalize technology name
            for person in people_names:
                has_tech = tech in person_technologies[person]
                row.append("✓" if has_tech else "✗")
            
            tech_table.append("| " + " | ".join(row) + " |")
        
        if len(tech_table) > 2:  # Only add if we have technologies
            if query_techs:
                formatted_output.append("\n### Other Technologies\n")
            formatted_output.extend(tech_table)
        
        # Add availability comparison table if focus area includes availability
        if comparison_results["focus_areas"]["availability"]:
            formatted_output.append("\n## Availability\n")
            
            # Create availability table
            avail_table = ["| Metric | " + " | ".join(people_names) + " |",
                        "| --- | " + " | ".join(["---"] * len(people_names)) + " |"]
            
            # Add rows for each availability metric
            metrics = [
                ("Current Project", lambda p, a: a.get("current_project") or "-"),
                ("Current Role", lambda p, a: a.get("current_role") or "-"),
                ("Expected End Date", lambda p, a: a.get("project_end_date") or "-"),
                ("Available", lambda p, a: "Yes" if a.get("is_available", False) else "No"),
                ("Availability Confidence", lambda p, a: f"{a.get('availability_confidence', 0)*100:.0f}%")
            ]
            
            for metric_name, value_func in metrics:
                row = [metric_name]
                for person in people_names:
                    avail = comparison_results["availability_comparison"].get(person, {})
                    row.append(value_func(person, avail))
                
                avail_table.append("| " + " | ".join(row) + " |")
            
            # Add availability table
            formatted_output.extend(avail_table)
        
        # Add common skills section
        if comparison_results["common_skills"]:
            formatted_output.append("\n## Common Skills\n")
            formatted_output.append("Skills shared by all compared individuals:")
            formatted_output.append(", ".join(comparison_results["common_skills"]))
        
        # Generate candidate rankings
        rankings = self.rank_candidates(comparison_results, query_intent)
        
        # Add ranking section
        formatted_output.append("\n## Candidate Ranking\n")
        
        # Create ranking table
        rank_table = ["| Rank | Person | Overall Score | Key Strengths |",
                    "| --- | --- | --- | --- |"]
        
        for i, ranking in enumerate(rankings, 1):
            person_name = ranking["name"]
            overall_score = ranking["scores"]["overall"]
            
            # Format score as percentage
            score_str = f"{overall_score * 100:.0f}%"
            
            # Get top 2 reasons/strengths
            reasons = ranking["reasons"][:2] if len(ranking["reasons"]) >= 2 else ranking["reasons"]
            
            # Format the row
            rank_table.append(f"| {i} | {person_name} | {score_str} | {'; '.join(reasons)} |")
        
        formatted_output.extend(rank_table)
        
        # Add more detailed breakdown of ranking
        formatted_output.append("\n### Ranking Criteria Breakdown\n")
        
        # Create criteria breakdown table
        criteria_table = ["| Person | " + " | ".join(crit.replace("_", " ").title() for crit in ["skill_proficiency", "project_experience", "technologies", "roles", "availability"] if crit in query_intent["comparison_criteria"]) + " |",
                        "| --- | " + " | ".join(["---"] * len([crit for crit in ["skill_proficiency", "project_experience", "technologies", "roles", "availability"] if crit in query_intent["comparison_criteria"]])) + " |"]
        
        for ranking in rankings:
            row = [ranking["name"]]
            
            for criterion in ["skill_proficiency", "project_experience", "technologies", "roles", "availability"]:
                if criterion in query_intent["comparison_criteria"]:
                    score = ranking["scores"].get(criterion, 0)
                    row.append(f"{score * 100:.0f}%")
            
            criteria_table.append("| " + " | ".join(row) + " |")
        
        formatted_output.extend(criteria_table)
        
        # Add recommendations section based on query focus
        formatted_output.append("\n## Recommendations\n")
        
        if skills_focus:
            recommendations = []
            top_person = rankings[0]["name"] if rankings else None
            
            if top_person:
                focus_text = ", ".join(skills_focus)
                recommendations.append(f"**For {focus_text}:** {top_person} is the strongest candidate based on skill proficiency and relevant project experience.")
            
            # Add recommendations for project types if specified
            if project_types:
                proj_type_text = ", ".join(project_types)
                
                # Find best person for each project type
                for project_type in project_types:
                    best_person = None
                    best_count = 0
                    
                    for person in people_names:
                        count = 0
                        for p in comparison_results["people"]:
                            if p["name"] == person:
                                for project in p.get("top_projects", []):
                                    content = (project.get("content", "") + " " + project.get("name", "")).lower()
                                    if project_type.lower() in content:
                                        count += 1
                        
                        if count > best_count:
                            best_count = count
                            best_person = person
                    
                    if best_person and best_count > 0:
                        recommendations.append(f"**For {project_type} projects:** {best_person} has the most experience with {best_count} relevant projects.")
            
            for recommendation in recommendations:
                formatted_output.append(f"- {recommendation}")
        else:
            # General recommendations based on ranking
            if rankings:
                formatted_output.append(f"- **Overall best match:** {rankings[0]['name']} (Score: {rankings[0]['scores']['overall'] * 100:.0f}%)")
                
                # Find specialists in different areas
                criteria_specialists = {}
                for criterion in ["skill_proficiency", "project_experience", "technologies", "roles", "availability"]:
                    if criterion in query_intent["comparison_criteria"]:
                        best_person = None
                        best_score = 0
                        
                        for ranking in rankings:
                            score = ranking["scores"].get(criterion, 0)
                            if score > best_score:
                                best_score = score
                                best_person = ranking["name"]
                        
                        if best_person and best_score > 0:
                            criteria_specialists[criterion] = (best_person, best_score)
                
                # Add specialist recommendations
                for criterion, (person, score) in criteria_specialists.items():
                    criterion_name = criterion.replace("_", " ").title()
                    formatted_output.append(f"- **Best for {criterion_name}:** {person} (Score: {score * 100:.0f}%)")
        
        return "\n".join(formatted_output)


   
    @staticmethod
    # Function to query projects and integrate with existing systems
    def query_resumes_improved(user_query):
        """
        Enhanced version of query_resumes with improved skill query handling.
        Uses the appropriate retrieval method based on query type.
        """
        try:
            # Initialize our contextual retrieval system
            retrieval_system = ContextualProjectRetrieval()
            
            # Initialize Bedrock LLM
            llm = ChatBedrock(
                model_id="anthropic.claude-3-haiku-20240307-v1:0",
                region_name="us-east-1",
                model_kwargs={
                    "temperature": 0.1,
                    "max_tokens": 1000,
                    "top_p": 0.9
                }
            )
            
            # Step 1: Detect query type with enhanced prompt for better skill detection
            query_type_prompt = """
            Determine if this is:
            1. A casual conversation/greeting
            2. A resume/candidate search query for finding people with specific skills
            3. A project-related query about past work and team assignments
            4. A comparison query (comparing multiple people)
            5. A general question unrelated to candidate/project search
            
            Examples of skills queries that should be classified as RESUME_QUERY:
            - "Find Java developers"
            - "Who knows Python"
            - "Developers with AWS skills"
            - "People with machine learning experience"
            - "Find React frontend developers"
            - "Who has experience with Kubernetes"
            
            Examples of project queries that should be classified as PROJECT_QUERY:
            - "Who worked on the CRM project?"
            - "Team members for mobile app"
            - "Projects completed in 2023"
            
            Respond only with "CONVERSATION", "RESUME_QUERY", "PROJECT_QUERY", "COMPARISON_QUERY", or "OUT_OF_SCOPE"
            
            Query: {query}
            """
            
            query_type_template = PromptTemplate(
                input_variables=["query"],
                template=query_type_prompt
            )
            
            query_type_chain = query_type_template | llm
            query_type_response = retrieval_system.invoke_bedrock_with_retry(query_type_chain, {"query": user_query})
            query_type = query_type_response.content.strip()
            print(f"Query type detected: {query_type}")
            
            # Step 2: Handle casual conversations
            if query_type == "CONVERSATION":
                conversation_reply_prompt = """
                Respond to this casual conversation in a friendly, professional manner.
                Keep your response short and crisp (maximum 2 lines).
                End with "How can I assist you in retrieving the company-wide information you're looking for?".
                
                Query: {query}
                """
                
                conversation_reply_template = PromptTemplate(
                    input_variables=["query"],
                    template=conversation_reply_prompt
                )
                
                reply_chain = conversation_reply_template | llm
                reply_response = retrieval_system.invoke_bedrock_with_retry(reply_chain, {"query": user_query})
                return reply_response.content.strip()
                
            # Step 3: Handle out-of-scope questions
            if query_type == "OUT_OF_SCOPE":
                return "I'm sorry, this question is out of my scope. Please try asking about resumes, candidates, skills, certifications, or projects to help you find the right talent."
            
            # Step 4: Handle comparison queries
            if query_type == "COMPARISON_QUERY":
                # Use the comparison handler
                return retrieval_system.handle_comparison_query(user_query, llm)
            
            # Step 5: Handle skill-related queries specifically (RESUME_QUERY)
            if query_type == "RESUME_QUERY":
                # For skill queries, use the specialized skill retrieval
                intention_prompt = """
                Analyze this user query and determine their true intention.
                What information are they actually looking for about candidates, projects, certifications
                
                Pay special attention to these categories:
                - Location queries (e.g., "Find developers in New York", "Who works in San Francisco?")
                - Project queries (e.g., "Who worked on the CRM project?", "Team members for mobile app")
                - Skills queries (e.g., "Find Python developers")
                - Certification queries (e.g., "Who has AWS certifications?")
                
                Categorize the query and describe the specific information needed.
                
                Query: {query}
                """
                
                intention_template = PromptTemplate(
                    input_variables=["query"],
                    template=intention_prompt
                )
                
                intention_chain = intention_template | llm
                intention_response = retrieval_system.invoke_bedrock_with_retry(intention_chain, {"query": user_query})
                query_intention = intention_response.content
                print(f"Query intention: {query_intention}")
                
                # Determine search type with stronger bias toward skills
                search_type = "all"
                if "skill" in query_intention.lower() or "experience" in query_intention.lower() or "know" in query_intention.lower():
                    search_type = "skills"
                    print("Set search_type to: skills")
                elif "certification" in query_intention.lower():
                    search_type = "certification"
                    print("Set search_type to: certification")
                elif "project" in query_intention.lower():
                    search_type = "projects"
                    print("Set search_type to: projects")
                
                # NEW: For skill-specific queries, ensure we don't redirect to project retrieval
                if search_type == "skills":
                    # Do NOT redirect to project contextual retrieval
                    # Instead, continue with skills-specific resume retrieval
                    # This is the key fix - we don't want to use contextual_project_retrieval for skills
                    candidate_names = retrieval_system.retrieve_resumes(user_query, search_type=search_type)
                    
                    if not candidate_names:
                        return "No candidates found with the requested skills. Would you like to broaden your search?"
                    
                    # Continue with skills-focused context retrieval
                    full_context = retrieval_system.get_all_resumes_from_store()
                    filtered_context = "\n\n---RESUME SEPARATOR---\n\n".join(
                        resume for resume in full_context.split("---RESUME SEPARATOR---") 
                        if any(name in resume for name in candidate_names)
                    )
                    
                    if not filtered_context:
                        return "No matching resumes found in the database."
                        
                    context_chunks = retrieval_system.chunk_context(filtered_context)
                    
                    # Define skills-specific prompt template
                    skills_template = """
                    You are an AI assistant analyzing resumes to find candidates with specific skills.
                    
                    Focus specifically on the skills mentioned in the query and provide information about:
                    1. The candidates' proficiency in these skills
                    2. Years of experience with these skills
                    3. Projects where they've applied these skills
                    4. Any certifications related to these skills
                    IMPORTANT: DO NOT include years of experience or duration information for any skills.
                    Do NOT include general project information unless directly related to the requested skills.
                    
                    Context (Chunk {chunk_num}/{total_chunks}):
                    {context}
                    
                    Query:
                    {query}
                    
                    User's Likely Intention: {intention}
                    
                    Provide a focused, skills-centric response that directly addresses the user's skills query.
                    List candidates in order of skill proficiency and experience.
                    """
                    
                    skills_prompt = PromptTemplate(
                        input_variables=["context", "query", "chunk_num", "total_chunks", "intention"],
                        template=skills_template
                    )
                    
                    skills_responses = []
                    for i, context_chunk in enumerate(context_chunks, 1):
                        chain = skills_prompt | llm
                        response = retrieval_system.invoke_bedrock_with_retry(chain, {
                            "context": context_chunk,
                            "query": user_query,
                            "chunk_num": i,
                            "total_chunks": len(context_chunks),
                            "intention": query_intention
                        })
                        
                        skills_responses.append(response.content)
                    
                    return "\n\n".join(skills_responses)
                elif search_type == "projects":
                    # Only for actual project queries within resume queries
                    return retrieval_system.query_projects_contextual(user_query, llm)
            
            # Step 6: Handle project-related queries
            if query_type == "PROJECT_QUERY":
                # Use the contextual retrieval system
                return retrieval_system.query_projects_contextual(user_query, llm)
            
            # Default fallback - process as a general resume query
            # Continue with the original intention analysis and resume retrieval process
            intention_prompt = """
            Analyze this user query and determine their true intention.
            What information are they actually looking for about candidates, projects, certifications
            
            Pay special attention to these categories:
            - Location queries (e.g., "Find developers in New York", "Who works in San Francisco?")
            - Project queries (e.g., "Who worked on the CRM project?", "Team members for mobile app")
            - Skills queries (e.g., "Find Python developers")
            - Certification queries (e.g., "Who has AWS certifications?")
            
            Categorize the query and describe the specific information needed.
            
            Query: {query}
            """
            
            intention_template = PromptTemplate(
                input_variables=["query"],
                template=intention_prompt
            )
            
            intention_chain = intention_template | llm
            intention_response = retrieval_system.invoke_bedrock_with_retry(intention_chain, {"query": user_query})
            query_intention = intention_response.content
            
            # Determine search type
            search_type = "all"
            if "project" in query_intention.lower():
                search_type = "projects"
            elif "certification" in query_intention.lower():
                search_type = "certification"
            elif "skill" in query_intention.lower():
                search_type = "skills"
            
            # Fall back to original retrieve_resumes function
            candidate_names = retrieval_system.retrieve_resumes(user_query, search_type=search_type)
            
            if not candidate_names:
                return "No matching candidates found. Would you like to broaden your search?"
            
            # Continue with original approach for context retrieval and response generation
            full_context = retrieval_system.get_all_resumes_from_store()
            filtered_context = "\n\n---RESUME SEPARATOR---\n\n".join(
                resume for resume in full_context.split("---RESUME SEPARATOR---") 
                if any(name in resume for name in candidate_names)
            )
            
            if not filtered_context:
                return "No matching resumes found in the database."
                
            context_chunks = retrieval_system.chunk_context(filtered_context)
            
            # Define prompt template
            template = """
            You are an AI assistant analyzing resumes and answering questions about candidates.
            
            SEARCH CONTEXT: {search_type}
            
            Analyze the resumes focusing on the following criteria:
            1. Location-based queries: Identify employees in specific geographic areas
            2. Project-based queries: Detail employee involvement in mentioned projects
            3. Skills matching: List candidates with specific technical skills
            4. Certification verification: Confirm and list professional certifications
            
            Key Analysis Points:
        
            - For project queries: List project name, role, duration, and key contributions
            - For skill queries: Highlight proficiency level and relevant experience
            - For certification queries: Include certification name, issuing body, and date
            
            Context (Chunk {chunk_num}/{total_chunks}):
            {context}
            
            Query:
            {query}
            
            User's Likely Intention: {intention}
            
            Provide a comprehensive, well-structured response that directly addresses the user's specific information need.
            """
            
            prompt = PromptTemplate(
                input_variables=["context", "query", "chunk_num", "total_chunks", "intention", "search_type"],
                template=template
            )
            
            all_responses = []
            
            for i, context_chunk in enumerate(context_chunks, 1):
                chain = prompt | llm
                response = retrieval_system.invoke_bedrock_with_retry(chain, {
                    "context": context_chunk,
                    "query": user_query,
                    "chunk_num": i,
                    "total_chunks": len(context_chunks),
                    "intention": query_intention,
                    "search_type": search_type
                })
                
                all_responses.append(response.content)
            
            return "\n\n".join(all_responses)
        
        except Exception as e:
            print(f"Error in query_resumes_improved: {e}")
            traceback.print_exc()
            return f"An error occurred while processing your query. Please try again later. Error details: {str(e)}"
    def contextual_project_retrieval(self, query: str, llm=None, top_k: int = 5, similarity_threshold: float = 0.3):
        """
        Advanced contextual project retrieval with comparison support.
        
        Args:
            query (str): The user query
            llm: Optional LLM for enhanced understanding
            top_k (int): Number of results to return
            similarity_threshold (float): Base threshold for semantic similarity
            
        Returns:
            dict: Results containing projects by person, by project name, and relevance
        """
        try:
            # Measure execution time
            start_time = time.time()
            
            # Step 1: Create project document store
            project_store = self.create_project_document_store()
            
            if not project_store['all_projects']:
                return {"results": [], "message": "No projects found in the database"}
            
            # Print available people for debugging
            print("Available people:", list(project_store['by_person'].keys()))
                
            # Step 2: Extract key terms from the query using both regex and embeddings
            query_terms = self.extract_key_terms(query)
            print(f"Extracted key terms from query: {query_terms}")
            
            # Step 3: Get query embedding for semantic similarity
            query_embedding = self.cached_embedding(query)
            
            # Check if this is a comparison query
            comparison_intent_score = self.detect_comparison_intent(query, query_embedding)
            
            if comparison_intent_score > 0.75:
                # This is a comparison query
                print(f"Detected comparison query (score: {comparison_intent_score:.2f})")
                comparison_result = self.handle_comparison_query(query, llm, top_k)
                return {"results": [], "message": comparison_result, "is_comparison": True}
            
            # Step 4: Dynamically build a knowledge graph of related terms from the corpus
            # Use cached term context vectors if available
            if self.term_context_vectors_cache is None:
                self.term_context_vectors_cache = self.build_term_context_vectors(project_store)
            term_context_vectors = self.term_context_vectors_cache
            
            # Step 5: Find related terms for all query terms
            expanded_terms = self.expand_query_terms(query_terms, term_context_vectors, query_embedding)
            print(f"Expanded query terms: {expanded_terms}")
            
            # Step 6: Initialize result containers
            results = {
                "by_person": [],
                "by_project": [],
                "by_relevance": [],
            }
            
            # Step 7: Determine query intent with embeddings
            person_intent_score, project_intent_score = self.determine_query_intent(query, query_embedding)
    
            print(f"Intent scores - Person: {person_intent_score:.2f}, Project: {project_intent_score:.2f}")
            
            # Step 8: Identify potential person/project
            all_persons = list(project_store['by_person'].keys())
            preprocessed_query = self.name_matching(query, all_persons)
            potential_person = self.find_potential_person(preprocessed_query, project_store, query_embedding, term_context_vectors)
            potential_project = None
            
            if not potential_person:
                potential_project = self.find_potential_project(query, project_store, query_embedding, term_context_vectors)
            
            # Step 9: Process based on intent and potential matches
            # Person-focused query - prioritize person-specific results
            if potential_person:
                print(f"Identified specific person: {potential_person}")
                
                # Get person's projects
                person_projects = project_store['by_person'].get(potential_person, [])
                
                # Score each project based on query relevance
                scored_projects = []
                for project in person_projects:
                    # Create project context for comparison
                    project_text = f"{project['project_name']} {project['project_role']} {project['content']}"
                    project_embedding = self.cached_embedding(project_text)
                    
                    # Calculate semantic similarity 
                    sem_similarity = self.cosine_similarity_calc([query_embedding], [project_embedding])
                    sem_sim_value = float(sem_similarity) if not hasattr(sem_similarity, 'shape') else float(sem_similarity[0][0])
                    
                    # Calculate term relevance
                    term_relevance = self.calculate_term_relevance(expanded_terms, project_text, term_context_vectors)
                    
                    # Combined score (weighted average)
                    combined_score = (sem_sim_value * 0.6) + (term_relevance * 0.4)
                    
                    # Add project regardless of threshold when person is specifically requested
                    project_copy = project.copy()
                    project_copy['similarity'] = combined_score
                    project_copy['semantic_similarity'] = sem_sim_value
                    project_copy['term_relevance'] = term_relevance
                    scored_projects.append(project_copy)
                
                # Sort by combined score
                scored_projects.sort(key=lambda x: x.get('similarity', 0), reverse=True)
                results["by_person"] = scored_projects
                
                # Clear other result types when specific person is requested
                results["by_project"] = []
                results["by_relevance"] = []
            
            # Project-focused query
            elif (project_intent_score > person_intent_score and project_intent_score > 0.7) or potential_project:
                print(f"Query appears to be about a specific project (intent score: {project_intent_score:.2f})")
                
                if potential_project:
                    print(f"Identified potential project: {potential_project}")
                    
                    # Get people working on this project
                    project_people = project_store['by_project'].get(potential_project, [])
                    
                    # Score each person based on query relevance
                    scored_people = []
                    for person in project_people:
                        # Calculate relevance to query
                        person_text = f"{person['name']} {person['project_role']} {person['content']}"
                        person_embedding = self.cached_embedding(person_text)
                        
                        # Calculate semantic similarity
                        sem_similarity = self.cosine_similarity_calc([query_embedding], [person_embedding])
                        sem_sim_value = float(sem_similarity) if not hasattr(sem_similarity, 'shape') else float(sem_similarity[0][0])
                        
                        # Calculate term relevance 
                        term_relevance = self.calculate_term_relevance(expanded_terms, person_text, term_context_vectors)
                        
                        # Combined score
                        combined_score = (sem_sim_value * 0.6) + (term_relevance * 0.4)
                        
                        # Add to results if above threshold
                        if combined_score > similarity_threshold * 0.7:
                            person_copy = person.copy()
                            person_copy['similarity'] = combined_score
                            scored_people.append(person_copy)
                    
                    # If no people met the threshold, include at least some results
                    if not scored_people and project_people:
                        for person in project_people:
                            person_text = f"{person['name']} {person['project_role']} {person['content']}"
                            person_embedding = self.cached_embedding(person_text)
                            
                            sem_similarity = self.cosine_similarity_calc([query_embedding], [person_embedding])
                            sem_sim_value = float(sem_similarity) if not hasattr(sem_similarity, 'shape') else float(sem_similarity[0][0])
                            
                            term_relevance = self.calculate_term_relevance(expanded_terms, person_text, term_context_vectors)
                            
                            combined_score = (sem_sim_value * 0.6) + (term_relevance * 0.4)
                            
                            person_copy = person.copy()
                            person_copy['similarity'] = combined_score
                            scored_people.append(person_copy)
                    
                    # Sort by relevance
                    scored_people.sort(key=lambda x: x.get('similarity', 0), reverse=True)
                    results["by_project"] = scored_people
                else:
                    # No specific project identified, search across all projects
                    
                    # Score all projects by relevance to query
                    project_people = []
                    for project_name, people in project_store['by_project'].items():
                        # Calculate project relevance
                        project_context = self.create_project_context(project_name, people)
                        project_embedding = self.cached_embedding(project_context)
                        project_similarity = self.cosine_similarity_calc([query_embedding], [project_embedding])
                        project_sim_value = float(project_similarity) if not hasattr(project_similarity, 'shape') else float(project_similarity[0][0])
                        
                        # Check term relevance for project name
                        name_term_relevance = self.calculate_term_relevance(expanded_terms, project_name, term_context_vectors)
                        
                        # Combined project score
                        project_score = (project_sim_value * 0.7) + (name_term_relevance * 0.3)
                        
                        # Only consider if above threshold
                        if project_score > similarity_threshold * 0.6:
                            # Get people for this project
                            for person in people:
                                person_copy = person.copy()
                                person_copy['similarity'] = project_score
                                project_people.append(person_copy)
                    
                    # If no project people met the threshold, include the top matches anyway
                    if not project_people:
                        all_project_people = []
                        for project_name, people in project_store['by_project'].items():
                            project_context = self.create_project_context(project_name, people)
                            project_embedding = self.cached_embedding(project_context)
                            project_similarity = self.cosine_similarity_calc([query_embedding], [project_embedding])
                            project_sim_value = float(project_similarity) if not hasattr(project_similarity, 'shape') else float(project_similarity[0][0])
                            
                            name_term_relevance = self.calculate_term_relevance(expanded_terms, project_name, term_context_vectors)
                            project_score = (project_sim_value * 0.7) + (name_term_relevance * 0.3)
                            
                            for person in people:
                                person_copy = person.copy()
                                person_copy['similarity'] = project_score
                                all_project_people.append(person_copy)
                        
                        all_project_people.sort(key=lambda x: x.get('similarity', 0), reverse=True)
                        project_people = all_project_people[:top_k]
                    
                    # Sort by relevance
                    project_people.sort(key=lambda x: x.get('similarity', 0), reverse=True)
                    results["by_project"] = project_people[:top_k]
                    
            # Person-focused query (by intent but no specific person identified)
            elif (person_intent_score > project_intent_score and person_intent_score > 0.7):
                print(f"Query appears to be about a person's projects (intent score: {person_intent_score:.2f})")
                
                # No specific person identified, search across all people
                
                # Score all persons by relevance to query
                person_projects = []
                for person_name, projects in project_store['by_person'].items():
                    # Calculate person relevance score
                    person_context = self.create_person_context(person_name, projects)
                    person_embedding = self.cached_embedding(person_context)
                    person_similarity = self.cosine_similarity_calc([query_embedding], [person_embedding])
                    person_sim_value = float(person_similarity) if not hasattr(person_similarity, 'shape') else float(person_similarity[0][0])
                    
                    # Only consider if above threshold
                    if person_sim_value > similarity_threshold * 0.6:
                        # Score each project for this person
                        for project in projects:
                            # Create project context
                            project_text = f"{project['project_name']} {project['project_role']} {project['content']}"
                            project_embedding = self.cached_embedding(project_text)
                            
                            # Calculate relevance scores
                            sem_similarity = self.cosine_similarity_calc([query_embedding], [project_embedding])
                            sem_sim_value = float(sem_similarity) if not hasattr(sem_similarity, 'shape') else float(sem_similarity[0][0])
                            
                            term_relevance = self.calculate_term_relevance(expanded_terms, project_text, term_context_vectors)
                            
                            # Combine scores with person relevance
                            combined_score = (sem_sim_value * 0.4) + (term_relevance * 0.3) + (person_sim_value * 0.3)
                            
                            # Add to results if above threshold
                            if combined_score > similarity_threshold * 0.7:
                                project_copy = project.copy()
                                project_copy['similarity'] = combined_score
                                person_projects.append(project_copy)
                
                # If no person projects met the threshold, include the top matches anyway
                if not person_projects:
                    # Collect all person-project combinations
                    all_person_projects = []
                    for person_name, projects in project_store['by_person'].items():
                        person_context = self.create_person_context(person_name, projects)
                        person_embedding = self.cached_embedding(person_context)
                        person_similarity = self.cosine_similarity_calc([query_embedding], [person_embedding])
                        person_sim_value = float(person_similarity) if not hasattr(person_similarity, 'shape') else float(person_similarity[0][0])
                        
                        for project in projects:
                            project_text = f"{project['project_name']} {project['project_role']} {project['content']}"
                            project_embedding = self.cached_embedding(project_text)
                            
                            sem_similarity = self.cosine_similarity_calc([query_embedding], [project_embedding])
                            sem_sim_value = float(sem_similarity) if not hasattr(sem_similarity, 'shape') else float(sem_similarity[0][0])
                            
                            term_relevance = self.calculate_term_relevance(expanded_terms, project_text, term_context_vectors)
                            
                            combined_score = (sem_sim_value * 0.4) + (term_relevance * 0.3) + (person_sim_value * 0.3)
                            
                            project_copy = project.copy()
                            project_copy['similarity'] = combined_score
                            all_person_projects.append(project_copy)
                    
                    # Take the top matches
                    all_person_projects.sort(key=lambda x: x.get('similarity', 0), reverse=True)
                    person_projects = all_person_projects[:top_k]
                
                # Sort by combined score
                person_projects.sort(key=lambda x: x.get('similarity', 0), reverse=True)
                results["by_person"] = person_projects[:top_k]
        
            else:
                print(f"Using general relevance search (person: {person_intent_score:.2f}, project: {project_intent_score:.2f})")
                
                # Search across all projects with expanded terms
                relevance_results = []
                all_scored_projects = []
                
                for project in project_store['all_projects']:
                    # Create comprehensive project text
                    project_text = f"{project['name']} worked on {project['project_name']} as {project['project_role']}. {project['content']}"
                    project_embedding = self.cached_embedding(project_text)
                    
                    # Calculate relevance scores
                    sem_similarity = self.cosine_similarity_calc([query_embedding], [project_embedding])
                    sem_sim_value = float(sem_similarity) if not hasattr(sem_similarity, 'shape') else float(sem_similarity[0][0])
                    
                    term_relevance = self.calculate_term_relevance(expanded_terms, project_text, term_context_vectors)
                    
                    # Combined score
                    combined_score = (sem_sim_value * 0.6) + (term_relevance * 0.4)
                    
                    # Store all scores for potential use
                    all_scored_projects.append((project, combined_score))
                    
                    # Add to results if above threshold
                    if combined_score > similarity_threshold * 0.7:
                        project_copy = project.copy()
                        project_copy['similarity'] = combined_score
                        relevance_results.append(project_copy)
                
                # If no projects met the threshold, include the top matches anyway
                if not relevance_results and all_scored_projects:
                    all_scored_projects.sort(key=lambda x: x[1], reverse=True)
                    top_projects = all_scored_projects[:top_k]
                    
                    for project, score in top_projects:
                        project_copy = project.copy()
                        project_copy['similarity'] = score
                        relevance_results.append(project_copy)
                
                # Sort by combined score
                relevance_results.sort(key=lambda x: x.get('similarity', 0), reverse=True)
                results["by_relevance"] = relevance_results[:top_k]
            
            # Step 10: Post-processing - Ensure person-specific queries only show that person
            if potential_person:
                # Filter all results to ensure only the specified person is included
                if results["by_person"]:
                    results["by_person"] = [p for p in results["by_person"] if p["name"] == potential_person]
                # Clear other result types for person-specific queries
                results["by_project"] = []
                results["by_relevance"] = []
            
            # Log execution time
            end_time = time.time()
            print(f"Query processed in {end_time - start_time:.2f} seconds")
            
            # Print the results before returning
            print(f"Results: {results}")
            
            return {"results": results, "message": "Success"}
        
        except Exception as e:
            print(f"Error in contextual_project_retrieval: {e}")
            traceback.print_exc()
            return {"results": [], "message": f"Error: {str(e)}"}
            