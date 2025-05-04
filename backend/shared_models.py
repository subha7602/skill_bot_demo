from typing import List, Optional
from pydantic import BaseModel, Field

class PersonalInfo(BaseModel):
    name: str
    job_title: str
    company: Optional[str] = None
    location: Optional[str] = None
    email: Optional[str] = None
    linkedin: Optional[str] = None

class DateRange(BaseModel):
    start_date: str = Field(..., description="Start date in YYYY-MM-DD format")
    end_date: Optional[str] = Field(None, description="End date in YYYY-MM-DD format")


class Certification(BaseModel):
    certification_name: str
    certificate_provider: str
    level: Optional[str] = None

class Project(BaseModel):
    project_role: str 
    project_name: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    responsibilities: List[str] = Field(default_factory=list) 

class Profile(BaseModel):
    name: str
    job_title: Optional[str] = None
    email: Optional[str] = None
    linkedin: Optional[str] = None
    location: Optional[str] = None
    executive_summary: Optional[str] = None
    skills: List[str] = Field(default_factory=list)
    projects: List[Project] = Field(default_factory=list)
    certifications: List[Certification] = Field(default_factory=list)
