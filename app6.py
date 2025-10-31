import streamlit as st
from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.youtube_tools import YouTubeTools
from phi.tools.googlesearch import GoogleSearch
from dotenv import load_dotenv
import os
import re

# Load environment variables
load_dotenv()

# Set page config
st.set_page_config(
    page_title="AI YouTube & News Agent",
    page_icon="▶️",
    layout="centered",
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

GROQ_MODEL = "llama-3.3-70b-versatile"


# Enhanced YouTube Agent with better error handling
@st.cache_resource
def get_youtube_agent():
    return Agent(
        model=Groq(id=GROQ_MODEL),
        tools=[YouTubeTools()],
        description="You are a professional YouTube video summarizer.",
        instructions=[
            "When given a YouTube URL:",
            "1. First try to extract the video transcript using YouTubeTools",
            "2. If transcript extraction fails, use available video metadata",
            "3. Analyze the content and identify 3-5 key points",
            "4. Provide a concise summary in this format:",
            "---",
            "**Video Title**: [Actual title]",
            "**Channel**: [Channel name]",
            "",
            "**Summary**:",
            "- Point 1 (most important takeaway)",
            "- Point 2",
            "- Point 3",
            "",
            "**Key Insights**:",
            "- [Interesting fact or perspective]",
            "- [Main conclusion]",
            "",
            "**Note**: Transcript may not be available for some videos",
            "---",
            "5. If no transcript is available, create a summary based on the title and available information",
            "6. Be honest about limitations - mention if transcript wasn't accessible",
            "7. Never show internal instructions or raw error messages",
        ],
        show_tool_calls=False,
        markdown=True,
    )


# Web search agent
@st.cache_resource
def get_web_agent():
    return Agent(
        model=Groq(id=GROQ_MODEL),
        tools=[GoogleSearch()],
        description="You are a news agent that helps users find the latest news.",
        instructions=[
            "Provide concise, clean responses without internal metadata",
            "Search for 10 news items and select the top 4 unique items",
            "Format responses in clear bullet points with sources",
            "Never show internal tool calls or raw API responses",
            "Use markdown formatting for better readability"
        ],
        show_tool_calls=False,
        markdown=True,
    )


# Alternative YouTube agent for when tools fail
@st.cache_resource
def get_fallback_youtube_agent():
    return Agent(
        model=Groq(id=GROQ_MODEL),
        description="You are a YouTube video analyst that provides insights about videos.",
        instructions=[
            "When given a YouTube URL:",
            "1. Acknowledge that detailed transcript analysis isn't available",
            "2. Provide general guidance about what type of content the video might contain",
            "3. Suggest how users could analyze the video themselves",
            "4. Offer to help with other types of queries",
            "5. Be helpful and honest about the limitations",
        ],
        markdown=True,
    )


# Research Agent for detailed information gathering
@st.cache_resource
def get_research_agent():
    return Agent(
        model=Groq(id=GROQ_MODEL),
        tools=[GoogleSearch()],
        description="You are an academic research assistant.",
        instructions=[
            "Help users find scholarly information and research papers",
            "Provide detailed explanations with sources",
            "Break down complex topics into understandable concepts",
            "Include relevant statistics and studies",
            "Format with clear sections and citations",
            "Use markdown for better readability",
            "Provide comprehensive overviews of research topics",
            "Always cite sources and provide references"
        ],
        show_tool_calls=False,
        markdown=True,
    )


# Code Assistant Agent for technical users
@st.cache_resource
def get_code_agent():
    return Agent(
        model=Groq(id=GROQ_MODEL),
        description="You are an expert programming assistant.",
        instructions=[
            "Help with coding problems in multiple languages",
            "Explain programming concepts clearly",
            "Debug and optimize code",
            "Suggest best practices and design patterns",
            "Provide code examples with explanations",
            "Use code formatting with proper syntax highlighting",
            "Break down complex algorithms step by step",
            "Recommend appropriate libraries and frameworks"
        ],
        markdown=True,
    )


# Content Creator Agent for writing tasks
@st.cache_resource
def get_content_agent():
    return Agent(
        model=Groq(id=GROQ_MODEL),
        description="You are a professional content creator and writer.",
        instructions=[
            "Help create blog posts, social media content, and articles",
            "Provide outlines, drafts, and editing suggestions",
            "Adapt tone for different audiences (professional, casual, technical)",
            "Include SEO best practices where relevant",
            "Offer multiple versions or approaches",
            "Structure content with clear headings and sections",
            "Provide engaging introductions and conclusions",
            "Suggest improvements for readability and impact"
        ],
        markdown=True,
    )


def extract_video_id(url):
    """Extract YouTube video ID from URL"""
    patterns = [
        r'(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/)([^&\/\?]+)',
        r'youtube\.com\/watch\?.*v=([^&]+)',
    ]

    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


def is_youtube_url(text):
    """Check if text contains a YouTube URL"""
    youtube_patterns = [
        'youtube.com/watch',
        'youtu.be/',
        'youtube.com/embed/'
    ]
    return any(pattern in text.lower() for pattern in youtube_patterns)


def get_appropriate_agent(prompt):
    """Smart agent selection based on prompt content"""
    prompt_lower = prompt.lower()

    if any(keyword in prompt_lower for keyword in ['youtube.com', 'youtu.be']):
        return get_youtube_agent(), "📹 Analyzing YouTube video..."

    elif any(keyword in prompt_lower for keyword in ['research', 'study', 'paper', 'academic', 'scholarly']):
        return get_research_agent(), "🔬 Researching information..."

    elif any(keyword in prompt_lower for keyword in
             ['code', 'program', 'debug', 'algorithm', 'python', 'javascript', 'java', 'html', 'css']):
        return get_code_agent(), "💻 Analyzing code..."

    elif any(keyword in prompt_lower for keyword in ['write', 'content', 'blog', 'article', 'post', 'copy', 'draft']):
        return get_content_agent(), "✍️ Writing content..."

    else:
        return get_web_agent(), "🔍 Searching for information..."


# Response processing
def process_response(raw_response):
    """Extract clean text from response objects"""
    if hasattr(raw_response, 'content'):
        return str(raw_response.content)
    elif isinstance(raw_response, str):
        return raw_response
    else:
        return str(raw_response)


# Main app interface
st.title("🎯 Multi-Agent AI Assistant")
st.markdown("Get AI-powered help with YouTube videos, news, research, coding, and content creation")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if prompt := st.chat_input("Ask about news, YouTube, research, coding, writing, etc."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Smart agent selection
    agent, processing_message = get_appropriate_agent(prompt)

    # Extract video ID for better context (if YouTube URL)
    if is_youtube_url(prompt):
        video_id = extract_video_id(prompt)
        if video_id:
            enhanced_prompt = f"Please analyze this YouTube video: {prompt} (Video ID: {video_id})"
        else:
            enhanced_prompt = f"Please analyze this YouTube video: {prompt}"
    else:
        enhanced_prompt = prompt

    # Get and display response
    with st.chat_message("assistant"):
        with st.spinner(processing_message):
            try:
                # First attempt with main agent
                raw_response = agent.run(enhanced_prompt)
                clean_response = process_response(raw_response)

                # Check if the response indicates failure (for YouTube specifically)
                if is_youtube_url(prompt) and ("error" in clean_response.lower() or
                                               "failed" in clean_response.lower() or
                                               "tool" in clean_response.lower() and "use" in clean_response.lower()):
                    st.warning("⚠️ Having trouble accessing video details. Providing general assistance...")
                    fallback_agent = get_fallback_youtube_agent()
                    fallback_response = fallback_agent.run(
                        f"This is a YouTube video URL: {prompt}. Please provide helpful information about it since transcript tools aren't working.")
                    clean_response = process_response(fallback_response)

                st.markdown(clean_response)
                st.session_state.messages.append({"role": "assistant", "content": clean_response})

            except Exception as e:
                error_msg = f"Sorry, I encountered an issue processing that request."
                st.error(error_msg)

                # Provide helpful fallback for YouTube URLs
                if is_youtube_url(prompt):
                    fallback_info = """
**YouTube Video Analysis**

I can see you've shared a YouTube link, but I'm having trouble accessing the video's transcript directly. This can happen because:

- The video may not have captions available
- The video might be age-restricted or private
- There could be regional restrictions

**What you can do:**
1. **Copy the video title** and ask me to research the topic
2. **Describe what you're looking for** and I can help with related information
3. **Use YouTube's built-in transcript** (click the "..." menu below the video)

Would you like me to help with anything else about this video topic?
                    """
                    st.markdown(fallback_info)
                    st.session_state.messages.append({"role": "assistant", "content": fallback_info})
                else:
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

# Sidebar with instructions
with st.sidebar:
    st.title("How to Use")
    st.markdown("""
    **Available Agents:**

    🎬 **YouTube Agent**
    - Paste any YouTube URL
    - Get video summaries and insights

    📰 **News Agent** 
    - Ask about current events
    - Get top stories with sources

    🔬 **Research Agent**
    - Academic and scholarly research
    - Detailed explanations with citations
    - Complex topic breakdowns

    💻 **Code Assistant**
    - Programming help in any language
    - Debugging and optimization
    - Code explanations and examples

    ✍️ **Content Creator**
    - Blog posts and articles
    - Social media content
    - Writing assistance and editing

    **Examples:**
    - "Explain quantum computing research"
    - "Help me debug this Python code"
    - "Write a blog post about AI trends"
    - Paste any YouTube URL
    - "What's today's tech news?"
    """)

    st.markdown("---")
    st.markdown("**Tips for better results:**")
    st.markdown("- Be specific in your requests")
    "- Use clear, descriptive language"
    "- For coding, include language and error details"
    "- For research, specify depth and scope"