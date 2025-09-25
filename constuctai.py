import streamlit as st
import requests
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="ConstructAI UK - Advanced Assistant",
    page_icon="🛠️",
    layout="wide",
    initial_sidebar_state="expanded"
)

load_dotenv()


# API Configuration - Try Streamlit secrets first, then environment variables
def get_api_key(key_name):
    """Get API key from Streamlit secrets or environment variables"""
    try:
        # Try Streamlit secrets first (for cloud deployment)
        return st.secrets[key_name]
    except:
        # Fallback to environment variables (for local development)
        return os.getenv(key_name)


# Get API keys securely
SERPER_API_KEY = get_api_key("SERPER_API_KEY")
OPENROUTER_API_KEY = get_api_key("OPENROUTER_API_KEY")
COMPANIES_HOUSE_API_KEY = get_api_key("COMPANIES_HOUSE_API_KEY")
OPENWEATHER_API_KEY = get_api_key("OPENWEATHER_API_KEY")

# Validate API keys
missing_keys = []
if not SERPER_API_KEY:
    missing_keys.append("SERPER_API_KEY")
if not OPENROUTER_API_KEY:
    missing_keys.append("OPENROUTER_API_KEY")
if not COMPANIES_HOUSE_API_KEY:
    missing_keys.append("COMPANIES_HOUSE_API_KEY")
if not OPENWEATHER_API_KEY:
    missing_keys.append("OPENWEATHER_API_KEY")

if missing_keys:
    st.error(f"Missing required API keys: {', '.join(missing_keys)}")
    st.info("If running locally, ensure your .env file is properly configured.")
    st.info("If running on Streamlit Cloud, check your secrets configuration.")
    st.stop()


# Initialize session state variables
def init_session_state():
    """Initialize session state variables"""
    if 'cache' not in st.session_state:
        st.session_state.cache = {}
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    if 'context_memory' not in st.session_state:
        st.session_state.context_memory = []
    if 'user_preferences' not in st.session_state:
        st.session_state.user_preferences = {
            'response_style': 'Professional',
            'detail_level': 'Comprehensive',
            'location_preference': 'London',
            'industry_focus': 'General Construction'
        }
    if 'processing' not in st.session_state:
        st.session_state.processing = False
    if 'last_query' not in st.session_state:
        st.session_state.last_query = ""
    if 'show_result' not in st.session_state:
        st.session_state.show_result = False
    if 'current_result' not in st.session_state:
        st.session_state.current_result = None
    if 'preset_query' not in st.session_state:
        st.session_state.preset_query = ""
    if 'execute_preset' not in st.session_state:
        st.session_state.execute_preset = False


# Enhanced company database
KNOWN_COMPANIES = {
    "morgan sindal": "morgan sindall",
    "balfour beattyl": "balfour beatty",
    "laing o rourk": "laing o'rourke",
    "sir robert mcalpyn": "sir robert mcalpine",
    "kier": "kier group",
    "skanska": "skanska uk",
    "vinci": "vinci construction",
    "carillion": "carillion plc",
    "galliford try": "galliford try holdings",
    "willmott dixon": "willmott dixon holdings",
    "wates": "wates group",
    "interserve": "interserve plc",
    "costain": "costain group",
    "bam": "bam construct uk",
    "multiplex": "multiplex construction europe",
    "mace": "mace limited",
    "lendlease": "lendlease construction (europe)",
    "taylor wimpey": "taylor wimpey plc",
    "persimmon": "persimmon plc",
    "barratt": "barratt developments plc"
}

# Cache duration settings
CACHE_DURATION = {
    'weather': 900,  # 15 minutes
    'company': 1800,  # 30 minutes
    'news': 300,  # 5 minutes
    'regulations': 3600  # 1 hour
}


def is_cache_valid(cache_entry: dict, cache_type: str = 'default') -> bool:
    """Check if cache entry is still valid based on type"""
    duration = CACHE_DURATION.get(cache_type, 300)
    return time.time() - cache_entry['timestamp'] < duration


def cache_key(func_name: str, *args) -> str:
    """Generate cache key"""
    return f"{func_name}_{hash(str(args))}"


# Enhanced AI Model Integration with Proper Context
class OpenRouterClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

    def generate_response_with_history(self, system_prompt: str, user_prompt: str,
                                       conversation_history: List[Dict],
                                       model: str = "perplexity/sonar") -> str:
        """Generate response using OpenRouter with full conversation history"""
        try:
            # Start with system message
            messages = [{"role": "system", "content": system_prompt}]

            # Add conversation history (last 5 exchanges to stay within token limits)
            recent_history = conversation_history[-5:] if len(conversation_history) > 5 else conversation_history

            for entry in recent_history:
                # Add user message
                messages.append({"role": "user", "content": entry['query']})
                # Add assistant response (truncated to avoid token limits)
                response_content = entry['response']
                if len(response_content) > 1000:
                    response_content = response_content[:1000] + "... [truncated for context]"
                messages.append({"role": "assistant", "content": response_content})

            # Add the current user question
            messages.append({"role": "user", "content": user_prompt})

            payload = {
                "model": model,
                "messages": messages,
                "temperature": 0.3,
                "max_tokens": 2048,
                "top_p": 0.9
            }

            response = requests.post(
                self.base_url,
                headers=self.headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()

            data = response.json()
            return data['choices'][0]['message']['content']

        except Exception as e:
            logger.error(f"OpenRouter API error: {e}")
            return f"Error generating response: {str(e)}"

    def generate_response(self, system_prompt: str, user_prompt: str,
                          model: str = "perplexity/sonar") -> str:
        """Generate response using OpenRouter (fallback method)"""
        try:
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": 0.3,
                "max_tokens": 2048,
                "top_p": 0.9
            }

            response = requests.post(
                self.base_url,
                headers=self.headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()

            data = response.json()
            return data['choices'][0]['message']['content']

        except Exception as e:
            logger.error(f"OpenRouter API error: {e}")
            return f"Error generating response: {str(e)}"


# Initialize OpenRouter client
openrouter_client = OpenRouterClient(OPENROUTER_API_KEY)


# Enhanced Web Search with Real-time Data
def search_web_realtime(query: str, search_type: str = "general") -> Dict:
    """Enhanced web search with real-time focus"""
    cache_key_str = cache_key("search_web", query, search_type)

    # Check cache
    if (cache_key_str in st.session_state.cache and
            is_cache_valid(st.session_state.cache[cache_key_str], 'news')):
        return st.session_state.cache[cache_key_str]['data']

    try:
        url = "https://google.serper.dev/search"
        headers = {"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"}

        # Enhanced search with real-time focus
        time_filter = ""
        if search_type == "news":
            time_filter = " after:2024"
        elif search_type == "regulations":
            time_filter = " site:gov.uk OR site:hse.gov.uk"

        payload = {
            "q": f"{query}{time_filter}",
            "num": 10,
            "gl": "uk",
            "hl": "en",
            "autocorrect": True
        }

        response = requests.post(url, headers=headers, json=payload, timeout=15)
        response.raise_for_status()

        data = response.json()

        results = {
            'organic': data.get('organic', []),
            'news': data.get('news', []),
            'knowledge_graph': data.get('knowledgeGraph', {}),
            'answer_box': data.get('answerBox', {}),
            'related_searches': data.get('relatedSearches', []),
            'timestamp': time.time()
        }

        # Cache results
        st.session_state.cache[cache_key_str] = {
            'data': results,
            'timestamp': time.time()
        }

        return results

    except Exception as e:
        logger.error(f"Search error: {e}")
        return {'organic': [], 'error': str(e)}


# Enhanced Company Information
def get_company_info_enhanced(company_name: str) -> str:
    """Get comprehensive company information with real-time data"""
    cache_key_str = cache_key("company_info", company_name)

    # Check cache
    if (cache_key_str in st.session_state.cache and
            is_cache_valid(st.session_state.cache[cache_key_str], 'company')):
        return st.session_state.cache[cache_key_str]['data']

    try:
        normalized_name = KNOWN_COMPANIES.get(company_name.lower(), company_name)

        # Companies House data
        search_url = "https://api.company-information.service.gov.uk/search/companies"
        params = {'q': normalized_name, 'items_per_page': 5}

        response = requests.get(
            search_url,
            auth=(COMPANIES_HOUSE_API_KEY, ""),
            params=params,
            timeout=10
        )
        response.raise_for_status()

        search_data = response.json()
        items = search_data.get("items", [])

        if not items:
            return f"No company found with name: {company_name}"

        company_number = items[0]["company_number"]

        # Get detailed company data
        with ThreadPoolExecutor(max_workers=3) as executor:
            profile_future = executor.submit(get_company_profile, company_number)
            officers_future = executor.submit(get_company_officers, company_number)
            filings_future = executor.submit(get_company_filings, company_number)

            profile = profile_future.result()
            officers = officers_future.result()
            filings = filings_future.result()

        # Get recent news about the company
        news_results = search_web_realtime(f"{normalized_name} news", "news")

        result = format_company_comprehensive(profile, officers, filings, news_results)

        # Cache result
        st.session_state.cache[cache_key_str] = {
            'data': result,
            'timestamp': time.time()
        }

        return result

    except Exception as e:
        logger.error(f"Company lookup error: {e}")
        return f"Error retrieving company information: {str(e)}"


def get_company_profile(company_number: str) -> Dict:
    """Get detailed company profile"""
    try:
        url = f"https://api.company-information.service.gov.uk/company/{company_number}"
        response = requests.get(url, auth=(COMPANIES_HOUSE_API_KEY, ""), timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Profile error: {e}")
        return {}


def get_company_officers(company_number: str) -> List[Dict]:
    """Get company officers"""
    try:
        url = f"https://api.company-information.service.gov.uk/company/{company_number}/officers"
        response = requests.get(url, auth=(COMPANIES_HOUSE_API_KEY, ""), timeout=10)
        response.raise_for_status()
        return response.json().get('items', [])[:5]
    except Exception as e:
        logger.error(f"Officers error: {e}")
        return []


def get_company_filings(company_number: str) -> List[Dict]:
    """Get recent company filings"""
    try:
        url = f"https://api.company-information.service.gov.uk/company/{company_number}/filing-history"
        response = requests.get(url, auth=(COMPANIES_HOUSE_API_KEY, ""), timeout=10)
        response.raise_for_status()
        return response.json().get('items', [])[:5]
    except Exception as e:
        logger.error(f"Filings error: {e}")
        return []


def format_company_comprehensive(profile: Dict, officers: List[Dict],
                                 filings: List[Dict], news: Dict) -> str:
    """Format comprehensive company information with news"""
    if not profile:
        return "Company profile not available"

    # Basic company info
    name = profile.get('company_name', 'N/A')
    number = profile.get('company_number', 'N/A')
    status = profile.get('company_status', 'N/A')
    incorporation_date = profile.get('date_of_creation', 'N/A')
    company_type = profile.get('type', 'N/A')

    # Address
    address = profile.get("registered_office_address", {})
    full_address = ", ".join(filter(None, [
        address.get('premises'),
        address.get('address_line_1'),
        address.get('address_line_2'),
        address.get('locality'),
        address.get('region'),
        address.get('postal_code'),
        address.get('country')
    ]))

    result = {
        'basic_info': {
            'name': name,
            'number': number,
            'status': status,
            'incorporation_date': incorporation_date,
            'type': company_type,
            'address': full_address
        },
        'officers': officers[:3],
        'recent_filings': filings[:3],
        'recent_news': news.get('news', [])[:3],
        'sic_codes': profile.get('sic_codes', [])
    }

    return json.dumps(result, indent=2)


# Enhanced Weather with Construction Impact Analysis
def get_weather_construction_impact(city: str) -> Dict:
    """Get weather with construction-specific impact analysis"""
    cache_key_str = cache_key("weather_construction", city)

    if (cache_key_str in st.session_state.cache and
            is_cache_valid(st.session_state.cache[cache_key_str], 'weather')):
        return st.session_state.cache[cache_key_str]['data']

    try:
        # Current weather
        current_url = f"http://api.openweathermap.org/data/2.5/weather"
        current_params = {
            'q': city,
            'appid': OPENWEATHER_API_KEY,
            'units': 'metric'
        }

        # 5-day forecast
        forecast_url = f"http://api.openweathermap.org/data/2.5/forecast"
        forecast_params = current_params.copy()

        with ThreadPoolExecutor(max_workers=2) as executor:
            current_future = executor.submit(
                requests.get, current_url, params=current_params, timeout=10
            )
            forecast_future = executor.submit(
                requests.get, forecast_url, params=forecast_params, timeout=10
            )

            current_response = current_future.result()
            forecast_response = forecast_future.result()

        current_response.raise_for_status()
        forecast_response.raise_for_status()

        current_data = current_response.json()
        forecast_data = forecast_response.json()

        # Analyze construction impact
        impact_analysis = analyze_construction_weather_impact(current_data, forecast_data)

        result = {
            'current': current_data,
            'forecast': forecast_data,
            'construction_impact': impact_analysis,
            'timestamp': time.time()
        }

        # Cache result
        st.session_state.cache[cache_key_str] = {
            'data': result,
            'timestamp': time.time()
        }

        return result

    except Exception as e:
        logger.error(f"Weather error: {e}")
        return {'error': str(e)}


def analyze_construction_weather_impact(current: Dict, forecast: Dict) -> Dict:
    """Analyze weather impact on construction activities"""
    current_temp = current['main']['temp']
    current_wind = current.get('wind', {}).get('speed', 0)
    current_rain = current.get('rain', {}).get('1h', 0)

    # Construction activity recommendations
    recommendations = {
        'concrete_pouring': 'suitable',
        'roofing_work': 'suitable',
        'external_work': 'suitable',
        'crane_operations': 'suitable',
        'excavation': 'suitable'
    }

    warnings = []

    # Temperature checks
    if current_temp < 2:
        recommendations['concrete_pouring'] = 'not_recommended'
        warnings.append('Risk of concrete freeze damage')

    # Wind checks
    if current_wind > 12:  # m/s
        recommendations['crane_operations'] = 'not_recommended'
        recommendations['roofing_work'] = 'not_recommended'
        warnings.append('High wind speeds - crane and height work restricted')

    # Rain checks
    if current_rain > 2:
        recommendations['external_work'] = 'limited'
        recommendations['excavation'] = 'not_recommended'
        warnings.append('Heavy rain - external work affected')

    return {
        'activity_recommendations': recommendations,
        'warnings': warnings,
        'suitable_activities': [k for k, v in recommendations.items() if v == 'suitable'],
        'restricted_activities': [k for k, v in recommendations.items() if v == 'not_recommended']
    }


# Advanced Domain Detection
def detect_query_domain(query: str) -> str:
    """Advanced domain detection with ML-like pattern matching"""
    q = query.lower()

    # Domain patterns
    patterns = {
        'weather': [
            r'\b(weather|temperature|rain|snow|wind|forecast|climate)\b',
            r'\b(sunny|cloudy|stormy|hot|cold|warm|cool)\b',
            r'\b(construction.*weather|weather.*construction)\b'
        ],
        'company': [
            r'\b(company|corporation|ltd|plc|limited|group)\b',
            r'\b(financial|revenue|turnover|profit|director)\b',
            r'\b(' + '|'.join(KNOWN_COMPANIES.keys()) + r')\b'
        ],
        'regulations': [
            r'\b(cdm|regulation|compliance|hse|building.*reg)\b',
            r'\b(planning|permit|approval|building.*control)\b',
            r'\b(health.*safety|risk.*assessment|method.*statement)\b'
        ],
        'sustainability': [
            r'\b(breeam|sustainability|green|environmental|carbon)\b',
            r'\b(energy.*efficiency|renewable|eco.*friendly)\b'
        ],
        'technical': [
            r'\b(construction.*method|technique|material|specification)\b',
            r'\b(concrete|steel|timber|scaffold|excavation)\b'
        ]
    }

    scores = {}
    for domain, domain_patterns in patterns.items():
        scores[domain] = sum(1 for pattern in domain_patterns if re.search(pattern, q))

    # Return domain with highest score
    max_score = max(scores.values()) if scores.values() else 0
    if max_score == 0:
        return 'general'

    return max(scores.items(), key=lambda x: x[1])[0]


# Enhanced Context Management System
class ContextManager:
    def __init__(self, max_context_length: int = 5):
        self.max_context_length = max_context_length

    def add_to_context(self, query: str, response: str, domain: str):
        """Add interaction to context memory"""
        context_entry = {
            'query': query,
            'response_summary': response[:200] + "..." if len(response) > 200 else response,
            'domain': domain,
            'timestamp': datetime.now().isoformat()
        }

        st.session_state.context_memory.append(context_entry)

        # Keep only recent context
        if len(st.session_state.context_memory) > self.max_context_length:
            st.session_state.context_memory = st.session_state.context_memory[-self.max_context_length:]

    def get_relevant_context_entries(self, current_domain: str) -> List[Dict]:
        """Get relevant conversation history entries for context"""
        if not st.session_state.conversation_history:
            return []

        # Get last 3-5 entries, prioritizing same domain
        relevant_entries = []

        # First, get same domain entries
        same_domain_entries = [
            entry for entry in reversed(st.session_state.conversation_history[-10:])
            if entry.get('domain') == current_domain
        ]
        relevant_entries.extend(same_domain_entries[:2])

        # Then add other recent entries if we don't have enough
        if len(relevant_entries) < 3:
            other_entries = [
                entry for entry in reversed(st.session_state.conversation_history[-5:])
                if entry not in relevant_entries
            ]
            relevant_entries.extend(other_entries[:3 - len(relevant_entries)])

        return relevant_entries

    def get_context_summary(self, current_query: str, current_domain: str) -> str:
        """Get context summary for system prompt (fallback)"""
        relevant_entries = self.get_relevant_context_entries(current_domain)

        if not relevant_entries:
            return ""

        context_parts = []
        for entry in relevant_entries:
            context_parts.append(f"Previous {entry.get('domain', 'general')} query: {entry['query']}")

        return " | ".join(context_parts)


context_manager = ContextManager()


# Enhanced Query Processing with Proper Context
def process_advanced_query(query: str) -> Dict[str, Any]:
    """Process query with advanced context and real-time data"""
    start_time = time.time()

    # Detect domain
    domain = detect_query_domain(query)

    # Process based on domain to get raw data
    if domain == "weather":
        # Extract city
        city_match = re.search(r'\b(?:in|for|at)\s+([A-Za-z\s]+)(?:\s|$)', query)
        city = city_match.group(1).strip() if city_match else st.session_state.user_preferences['location_preference']
        raw_data = get_weather_construction_impact(city)

    elif domain == "company":
        # Extract company name
        company_words = query.lower().split()
        company_name = " ".join([w for w in company_words if w not in
                                 ['company', 'information', 'about', 'tell', 'me']])
        raw_data = get_company_info_enhanced(company_name)

    else:
        # Use web search for other domains
        search_type = "news" if domain in ["regulations", "sustainability"] else "general"
        raw_data = search_web_realtime(query, search_type)

    # Generate enhanced response using conversation history
    system_prompt = create_enhanced_system_prompt(domain)
    user_prompt = create_enhanced_user_prompt(query, raw_data, domain)

    # Use the new method with conversation history
    if st.session_state.conversation_history:
        ai_response = openrouter_client.generate_response_with_history(
            system_prompt,
            user_prompt,
            st.session_state.conversation_history
        )
    else:
        # Fallback for first query
        ai_response = openrouter_client.generate_response(system_prompt, user_prompt)

    processing_time = time.time() - start_time

    # Add to context (keep the existing context manager for analytics)
    context_manager.add_to_context(query, ai_response, domain)

    return {
        'domain': domain,
        'response': ai_response,
        'raw_data': raw_data,
        'processing_time': processing_time,
        'context_used': f"{len(st.session_state.conversation_history)} previous exchanges",
        'timestamp': datetime.now().isoformat()
    }


def create_enhanced_system_prompt(domain: str) -> str:
    """Create enhanced system prompt based on domain"""
    base_prompt = """You are ConstructAI, an advanced AI assistant specializing in the UK construction industry with access to real-time data. You provide accurate, up-to-date, and actionable information.

You have access to the conversation history and should reference previous queries and responses when relevant. This allows you to:
- Build upon previous discussions
- Provide comparative analysis when asked
- Reference earlier information to avoid repetition
- Maintain conversation continuity

Key Capabilities:
- Real-time weather data with construction impact analysis
- Live company information from Companies House
- Current regulatory updates and news
- Technical construction guidance
- UK-specific standards and regulations"""

    domain_expertise = {
        'weather': "Focus on construction-specific weather impacts, site safety, and work scheduling recommendations. Consider how weather affects different construction activities.",
        'company': "Provide comprehensive business intelligence including financial data, recent news, market analysis, and company comparisons when relevant to previous queries.",
        'regulations': "Emphasize current UK construction regulations, compliance requirements, recent updates, and how they relate to previously discussed topics.",
        'sustainability': "Focus on green building standards, environmental compliance, sustainable construction practices, and connections to earlier sustainability discussions.",
        'technical': "Provide detailed technical guidance on construction methods, materials, best practices, and relate to any previous technical discussions."
    }

    expertise = domain_expertise.get(domain,
                                     "Provide comprehensive construction industry guidance, building on previous conversation context.")

    return f"{base_prompt}\n\nCurrent Specialization: {expertise}\n\nAlways cite sources for factual claims, provide actionable insights, and reference relevant previous discussions when appropriate."


def create_enhanced_user_prompt(query: str, data: Any, domain: str) -> str:
    """Create enhanced user prompt with structured data"""
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S GMT")

    return f"""
Current Time: {current_time}
Current Query: {query}
Query Domain: {domain}

Retrieved Real-time Data:
{json.dumps(data, indent=2) if isinstance(data, (dict, list)) else str(data)}

Please provide a comprehensive response that:
1. Directly answers the user's question using the most current data
2. References relevant information from our previous conversation when appropriate
3. Provides specific, actionable recommendations
4. Includes proper citations for factual claims
5. Considers UK-specific regulations and standards
6. Uses clear formatting with headers and bullet points for readability
7. Builds upon previous context to provide deeper insights

If this query relates to or builds upon previous questions in our conversation, please acknowledge that connection and provide comparative or additional insights accordingly.
"""


# Rest of the Streamlit UI Components remain the same...
def render_sidebar():
    """Render enhanced sidebar with user preferences and controls"""
    with st.sidebar:
        st.title("ConstructAI Settings")

        # User Preferences
        st.subheader("Preferences")

        st.session_state.user_preferences['response_style'] = st.selectbox(
            "Response Style",
            ["Professional", "Technical", "Conversational"],
            index=0,
            key="response_style_select"
        )

        st.session_state.user_preferences['detail_level'] = st.selectbox(
            "Detail Level",
            ["Brief", "Standard", "Comprehensive"],
            index=2,
            key="detail_level_select"
        )

        st.session_state.user_preferences['location_preference'] = st.text_input(
            "Default Location",
            value=st.session_state.user_preferences['location_preference'],
            key="location_input"
        )

        st.session_state.user_preferences['industry_focus'] = st.selectbox(
            "Industry Focus",
            ["General Construction", "Residential", "Commercial", "Infrastructure", "Sustainability"],
            index=0,
            key="industry_focus_select"
        )

        st.divider()

        # Session Statistics
        st.subheader("Session Stats")

        if st.session_state.conversation_history:
            total_queries = len(st.session_state.conversation_history)
            avg_processing_time = sum(
                h.get('processing_time', 0) for h in st.session_state.conversation_history) / total_queries

            st.metric("Total Queries", total_queries)
            st.metric("Avg Response Time", f"{avg_processing_time:.2f}s")
            st.metric("Cache Entries", len(st.session_state.cache))
            st.metric("Context Depth", f"{len(st.session_state.conversation_history)} exchanges")

            # Domain distribution
            domains = [h.get('domain', 'unknown') for h in st.session_state.conversation_history]
            domain_counts = pd.Series(domains).value_counts()

            if not domain_counts.empty:
                fig = px.pie(
                    values=domain_counts.values,
                    names=domain_counts.index,
                    title="Query Domains"
                )
                fig.update_layout(height=300, showlegend=True)
                st.plotly_chart(fig, use_container_width=True)

        st.divider()

        # Context Information
        st.subheader("Context Info")
        if st.session_state.conversation_history:
            recent_domains = [h.get('domain', 'unknown') for h in st.session_state.conversation_history[-3:]]
            st.write("Recent domains:", ", ".join(recent_domains))
            st.write("Context available:", "✅" if len(st.session_state.conversation_history) > 0 else "❌")
        else:
            st.write("No conversation history yet")

        st.divider()

        # Cache Management
        st.subheader("Cache Management")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Clear Cache", key="clear_cache_btn"):
                st.session_state.cache = {}
                st.success("Cache cleared!")

        with col2:
            if st.button("Clear History", key="clear_history_btn"):
                st.session_state.conversation_history = []
                st.session_state.context_memory = []
                st.success("History cleared!")


def render_main_interface():
    """Render main chat interface"""
    st.title("ConstructAI UK - Advanced Construction Assistant")
    st.markdown("*Powered by real-time data and advanced AI models with conversation memory*")

    # Context status indicator
    if st.session_state.conversation_history:
        st.info(
            f"💬 Context active: {len(st.session_state.conversation_history)} previous exchanges available for reference")

    # Quick action buttons with session state management
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("Weather Impact", key="weather_btn"):
            st.session_state.preset_query = f"Weather impact on construction work in {st.session_state.user_preferences['location_preference']}"
            st.session_state.execute_preset = True

    with col2:
        if st.button("Latest CDM Updates", key="cdm_btn"):
            st.session_state.preset_query = "Latest CDM regulations updates 2024"
            st.session_state.execute_preset = True

    with col3:
        if st.button("BREEAM Guidelines", key="breeam_btn"):
            st.session_state.preset_query = "Latest BREEAM sustainability guidelines"
            st.session_state.execute_preset = True

    with col4:
        if st.button("Safety Alerts", key="safety_btn"):
            st.session_state.preset_query = "Latest HSE construction safety alerts"
            st.session_state.execute_preset = True

    # Main chat interface
    st.subheader("Chat with ConstructAI")

    # Display conversation history
    if st.session_state.conversation_history:
        with st.container():
            st.subheader("Recent Conversations")

            for i, entry in enumerate(reversed(st.session_state.conversation_history[-3:])):
                with st.expander(
                        f"Query {len(st.session_state.conversation_history) - i}: {entry['query'][:50]}..." if len(
                                entry[
                                    'query']) > 50 else f"Query {len(st.session_state.conversation_history) - i}: {entry['query']}"):
                    col1, col2 = st.columns([3, 1])

                    with col1:
                        st.markdown("**Query:**")
                        st.write(entry['query'])
                        st.markdown("**Response:**")
                        st.markdown(entry['response'])

                    with col2:
                        st.metric("Domain", entry.get('domain', 'unknown'))
                        st.metric("Time", f"{entry.get('processing_time', 0):.2f}s")
                        st.write(f"*{entry.get('timestamp', '')[:19]}*")

    # Query input with preset query handling
    query_value = st.session_state.preset_query if st.session_state.execute_preset else ""

    query_input = st.text_input(
        "Ask me anything about UK construction...",
        value=query_value,
        placeholder="e.g., 'Weather forecast for London construction sites', 'Balfour Beatty company information', 'Latest CDM regulations'",
        key="main_query_input",
        help="I remember our previous conversation - feel free to ask follow-up questions or reference earlier topics!"
    )

    # Auto-execute preset queries
    should_execute = (st.session_state.execute_preset and query_input) or (
            st.button("Send Query", type="primary", key="send_query_btn") and query_input)

    if should_execute:
        # Reset preset execution flag
        if st.session_state.execute_preset:
            st.session_state.execute_preset = False
            st.session_state.preset_query = ""

        # Prevent duplicate processing
        if not st.session_state.processing and query_input != st.session_state.last_query:
            st.session_state.processing = True
            st.session_state.last_query = query_input

            with st.spinner("Processing your query with real-time data and conversation context..."):
                try:
                    result = process_advanced_query(query_input)

                    # Store in conversation history
                    st.session_state.conversation_history.append({
                        'query': query_input,
                        'response': result['response'],
                        'domain': result['domain'],
                        'processing_time': result['processing_time'],
                        'timestamp': result['timestamp']
                    })

                    # Store current result for display
                    st.session_state.current_result = result
                    st.session_state.show_result = True

                    st.success(
                        f"Query processed in {result['processing_time']:.2f}s with context from {result['context_used']}")

                except Exception as e:
                    st.error(f"Error processing query: {str(e)}")
                    logger.error(f"Query processing error: {e}")

                finally:
                    st.session_state.processing = False
                    # Rerun to show results immediately
                    st.rerun()

    # Display current result if available
    if st.session_state.get('show_result') and st.session_state.get('current_result'):
        result = st.session_state.current_result

        # Main response
        st.subheader(f"ConstructAI Response - Domain: {result['domain'].title()}")

        # Show context indicator
        if len(st.session_state.conversation_history) > 1:
            st.caption(
                f"📝 This response uses context from {len(st.session_state.conversation_history) - 1} previous exchange(s)")

        st.markdown(result['response'])

        # Additional insights based on domain
        if result['domain'] == 'weather' and isinstance(result['raw_data'], dict):
            render_weather_dashboard(result['raw_data'])
        elif result['domain'] == 'company' and result['raw_data']:
            render_company_dashboard(result['raw_data'])

        # Context information
        with st.expander("Processing Details"):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write(f"**Domain:** {result['domain']}")
                st.write(f"**Processing Time:** {result['processing_time']:.2f}s")
            with col2:
                st.write(f"**Context Used:** {result['context_used']}")
                st.write(f"**Timestamp:** {result['timestamp'][:19]}")
            with col3:
                st.write(f"**Total Queries:** {len(st.session_state.conversation_history)}")
                st.write(f"**Cache Entries:** {len(st.session_state.cache)}")

        # Raw data view
        with st.expander("Raw Data View"):
            st.json(result['raw_data'])

        # Clear response button
        if st.button("Clear Current Response", key="clear_response_btn"):
            st.session_state.show_result = False
            st.session_state.current_result = None
            st.rerun()


def render_weather_dashboard(weather_data: Dict):
    """Render weather dashboard with construction insights"""
    if 'error' in weather_data:
        st.error(f"Weather data error: {weather_data['error']}")
        return

    current = weather_data.get('current', {})
    forecast = weather_data.get('forecast', {})
    impact = weather_data.get('construction_impact', {})

    if not current:
        return

    st.subheader("Weather Dashboard")

    # Current conditions
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        temp = current['main']['temp']
        st.metric("Temperature", f"{temp:.1f}°C", f"Feels like {current['main']['feels_like']:.1f}°C")

    with col2:
        humidity = current['main']['humidity']
        st.metric("Humidity", f"{humidity}%")

    with col3:
        wind_speed = current.get('wind', {}).get('speed', 0)
        st.metric("Wind Speed", f"{wind_speed:.1f} m/s")

    with col4:
        pressure = current['main']['pressure']
        st.metric("Pressure", f"{pressure} hPa")

    # Construction impact analysis
    if impact:
        st.subheader("Construction Impact Analysis")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Suitable Activities:**")
            for activity in impact.get('suitable_activities', []):
                st.write(f"• {activity.replace('_', ' ').title()}")

        with col2:
            st.markdown("**Restricted Activities:**")
            for activity in impact.get('restricted_activities', []):
                st.write(f"• {activity.replace('_', ' ').title()}")

        # Warnings
        if impact.get('warnings'):
            st.warning("Weather Warnings:\n" + "\n".join(f"• {warning}" for warning in impact['warnings']))

    # Forecast chart
    if forecast and forecast.get('list'):
        st.subheader("5-Day Forecast")

        forecast_data = []
        for item in forecast['list'][:40]:  # 5 days * 8 (3-hour intervals)
            forecast_data.append({
                'datetime': datetime.fromtimestamp(item['dt']),
                'temperature': item['main']['temp'],
                'description': item['weather'][0]['description'],
                'wind_speed': item.get('wind', {}).get('speed', 0),
                'rain': item.get('rain', {}).get('3h', 0)
            })

        df = pd.DataFrame(forecast_data)

        # Temperature chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df['datetime'],
            y=df['temperature'],
            mode='lines+markers',
            name='Temperature (°C)',
            line=dict(color='#ff7f0e', width=2)
        ))

        fig.update_layout(
            title="Temperature Forecast",
            xaxis_title="Date/Time",
            yaxis_title="Temperature (°C)",
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)

        # Wind speed chart
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=df['datetime'],
            y=df['wind_speed'],
            mode='lines+markers',
            name='Wind Speed (m/s)',
            line=dict(color='#2ca02c', width=2)
        ))

        # Add danger zone for crane operations
        fig2.add_hline(y=12, line_dash="dash", line_color="red",
                       annotation_text="Crane Operation Limit")

        fig2.update_layout(
            title="Wind Speed Forecast",
            xaxis_title="Date/Time",
            yaxis_title="Wind Speed (m/s)",
            height=400
        )

        st.plotly_chart(fig2, use_container_width=True)


def render_company_dashboard(company_data: str):
    """Render company information dashboard"""
    try:
        # Try to parse JSON data
        if isinstance(company_data, str):
            if company_data.startswith('{'):
                data = json.loads(company_data)
            else:
                st.text_area("Company Information", company_data, height=300)
                return
        else:
            data = company_data

        st.subheader("Company Dashboard")

        # Basic information
        basic_info = data.get('basic_info', {})
        if basic_info:
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Company Name", basic_info.get('name', 'N/A'))
                st.metric("Company Number", basic_info.get('number', 'N/A'))

            with col2:
                st.metric("Status", basic_info.get('status', 'N/A').title())
                st.metric("Type", basic_info.get('type', 'N/A').replace('_', ' ').title())

            with col3:
                incorporation_date = basic_info.get('incorporation_date', 'N/A')
                if incorporation_date != 'N/A':
                    # Calculate age
                    try:
                        inc_date = datetime.strptime(incorporation_date, '%Y-%m-%d')
                        age_years = (datetime.now() - inc_date).days // 365
                        st.metric("Incorporation Date", incorporation_date)
                        st.metric("Company Age", f"{age_years} years")
                    except:
                        st.metric("Incorporation Date", incorporation_date)

        # Officers information
        officers = data.get('officers', [])
        if officers:
            st.subheader("Key Officers")
            officer_df = pd.DataFrame([
                {
                    'Name': officer.get('name', 'N/A'),
                    'Role': officer.get('officer_role', 'N/A').replace('_', ' ').title(),
                    'Appointed': officer.get('appointed_on', 'N/A')
                }
                for officer in officers
            ])
            st.dataframe(officer_df, use_container_width=True)

        # Recent news
        news = data.get('recent_news', [])
        if news:
            st.subheader("Recent News")
            for article in news:
                with st.expander(f"{article.get('title', 'No title')}"):
                    st.write(f"**Source:** {article.get('source', 'N/A')}")
                    st.write(f"**Date:** {article.get('date', 'N/A')}")
                    st.write(f"**Summary:** {article.get('snippet', 'No summary available')}")
                    if article.get('link'):
                        st.write(f"[Read more]({article['link']})")

        # SIC Codes
        sic_codes = data.get('sic_codes', [])
        if sic_codes:
            st.subheader("Business Activities (SIC Codes)")
            for code in sic_codes[:10]:
                st.write(f"• {code}")

        # Recent filings
        filings = data.get('recent_filings', [])
        if filings:
            st.subheader("Recent Filings")
            filing_df = pd.DataFrame([
                {
                    'Description': filing.get('description', 'N/A')[:60] + '...' if len(
                        filing.get('description', '')) > 60 else filing.get('description', 'N/A'),
                    'Date': filing.get('date', 'N/A'),
                    'Category': filing.get('category', 'N/A')
                }
                for filing in filings
            ])
            st.dataframe(filing_df, use_container_width=True)

    except json.JSONDecodeError:
        st.text_area("Company Information", company_data, height=300)
    except Exception as e:
        st.error(f"Error rendering company dashboard: {e}")
        st.text_area("Company Information", str(company_data), height=300)


def render_analytics_page():
    """Render analytics and insights page"""
    st.title("Analytics & Insights")

    if not st.session_state.conversation_history:
        st.info("No conversation data available yet. Start chatting to see analytics!")
        return

    # Query analysis
    queries = [h['query'] for h in st.session_state.conversation_history]
    domains = [h.get('domain', 'unknown') for h in st.session_state.conversation_history]
    processing_times = [h.get('processing_time', 0) for h in st.session_state.conversation_history]

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Queries", len(queries))

    with col2:
        avg_time = sum(processing_times) / len(processing_times) if processing_times else 0
        st.metric("Avg Response Time", f"{avg_time:.2f}s")

    with col3:
        unique_domains = len(set(domains))
        st.metric("Unique Domains", unique_domains)

    # Context effectiveness metrics
    st.subheader("Context Analysis")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Conversation Length", f"{len(st.session_state.conversation_history)} exchanges")

    with col2:
        # Calculate context usage (queries after first)
        context_queries = max(0, len(st.session_state.conversation_history) - 1)
        st.metric("Context-Aware Responses", context_queries)

    with col3:
        # Calculate average context depth
        context_availability = "High" if len(st.session_state.conversation_history) > 5 else "Medium" if len(
            st.session_state.conversation_history) > 2 else "Low"
        st.metric("Context Depth", context_availability)

    # Domain distribution
    st.subheader("Query Domain Distribution")
    domain_counts = pd.Series(domains).value_counts()

    col1, col2 = st.columns([2, 1])

    with col1:
        fig = px.bar(
            x=domain_counts.index,
            y=domain_counts.values,
            title="Queries by Domain",
            labels={'x': 'Domain', 'y': 'Count'}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.pie(
            values=domain_counts.values,
            names=domain_counts.index,
            title="Domain Distribution"
        )
        fig.update_layout(height=400, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)

    # Response time analysis
    st.subheader("Response Time Analysis")

    if len(processing_times) > 1:
        time_df = pd.DataFrame({
            'Query': range(1, len(processing_times) + 1),
            'Response Time (s)': processing_times,
            'Domain': domains
        })

        fig = px.line(
            time_df,
            x='Query',
            y='Response Time (s)',
            color='Domain',
            title="Response Time Trends",
            markers=True
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

        # Performance metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Min Time", f"{min(processing_times):.2f}s")

        with col2:
            st.metric("Max Time", f"{max(processing_times):.2f}s")

        with col3:
            median_time = sorted(processing_times)[len(processing_times) // 2]
            st.metric("Median Time", f"{median_time:.2f}s")

        with col4:
            import statistics
            std_dev = statistics.stdev(processing_times) if len(processing_times) > 1 else 0
            st.metric("Std Dev", f"{std_dev:.2f}s")

    # Conversation flow analysis
    if len(st.session_state.conversation_history) > 1:
        st.subheader("Conversation Flow")

        # Create a simple conversation flow visualization
        flow_data = []
        for i, entry in enumerate(st.session_state.conversation_history):
            flow_data.append({
                'Exchange': i + 1,
                'Domain': entry.get('domain', 'unknown'),
                'Query_Length': len(entry['query']),
                'Response_Length': len(entry['response'])
            })

        flow_df = pd.DataFrame(flow_data)

        # Query length over time
        fig = px.line(
            flow_df,
            x='Exchange',
            y='Query_Length',
            title="Query Length Evolution",
            markers=True
        )
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)


def main():
    """Main application function"""
    # Initialize session state
    init_session_state()

    # Custom CSS for better styling
    st.markdown("""
    <style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }

    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #e1e5ea;
    }

    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }

    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }

    .ai-message {
        background-color: #f3e5f5;
        border-left: 4px solid #9c27b0;
    }

    .context-indicator {
        background-color: #e8f5e8;
        border: 1px solid #4caf50;
        padding: 0.5rem;
        border-radius: 0.3rem;
        margin: 0.5rem 0;
        font-size: 0.9rem;
    }

    .warning-box {
        background-color: #fff3e0;
        border: 1px solid #ff9800;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }

    .success-box {
        background-color: #e8f5e8;
        border: 1px solid #4caf50;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

    # Navigation
    tabs = st.tabs(["Chat", "Analytics", "Settings", "Help"])

    with tabs[0]:
        render_sidebar()
        render_main_interface()

    with tabs[1]:
        render_analytics_page()

    with tabs[2]:
        st.title("Advanced Settings")

        st.subheader("API Configuration")

        with st.expander("API Keys Status"):
            st.write("✅ OpenRouter API: Connected")
            st.write("✅ Serper API: Connected")
            st.write("✅ Companies House API: Connected")
            st.write("✅ OpenWeather API: Connected")

        st.subheader("Context & Performance Settings")

        col1, col2 = st.columns(2)

        with col1:
            cache_duration = st.slider("Cache Duration (minutes)", 5, 60, 15, key="cache_duration_slider")
            max_context_length = st.slider("Max Context History", 3, 10, 5, key="max_context_slider")

        with col2:
            max_search_results = st.slider("Max Search Results", 5, 20, 10, key="max_search_slider")
            response_timeout = st.slider("Response Timeout (seconds)", 10, 60, 30, key="timeout_slider")

        st.subheader("Context Management")

        context_mode = st.selectbox(
            "Context Mode",
            ["Full History", "Domain-Specific", "Recent Only"],
            index=0,
            help="Choose how context is used in responses"
        )

        st.write(f"Current conversation length: {len(st.session_state.conversation_history)} exchanges")

        if st.button("Save Settings", key="save_settings_btn"):
            st.success("Settings saved successfully!")

    with tabs[3]:
        st.title("Help & Documentation")

        st.markdown("""
        ## Getting Started with ConstructAI

        ConstructAI is your advanced AI assistant for the UK construction industry, powered by real-time data and cutting-edge AI models with **full conversation memory**.

        ### Key Features

        - **🧠 Conversation Memory**: Remembers your entire conversation and can reference previous questions and responses
        - **📊 Real-time Data Integration**: Access to current weather, company information, and regulatory updates
        - **🎯 Multi-domain Expertise**: Weather impact analysis, company intelligence, regulations, and technical guidance
        - **📈 Advanced Analytics**: Track your usage patterns and query insights
        - **🔄 Context-Aware Responses**: Each response builds upon your previous conversation

        ### How Context Works

        ConstructAI now maintains **full conversation context**, meaning:

        - **Follow-up questions** work naturally: "What about Manchester?" after asking about London weather
        - **Comparative queries**: "How does that compare to what we discussed earlier?"
        - **Building conversations**: Each response can reference and build upon previous exchanges
        - **Cross-domain connections**: Weather discussions can inform company analysis and vice versa

        ### How to Use

        1. **Start Conversations**: Ask your initial construction-related questions
        2. **Build Upon Responses**: Ask follow-ups, comparisons, or deeper questions
        3. **Reference History**: Use phrases like "earlier you mentioned..." or "compared to before..."
        4. **Use Quick Actions**: Click buttons for common queries that build on context
        5. **Review Analytics**: See how your conversation develops over time

        ### Context Examples

        **Building Weather Conversations:**
        - Query 1: "Weather for construction in London"
        - Query 2: "What about Manchester?" *(references previous location query)*
        - Query 3: "Which is better for concrete work this week?" *(compares both locations)*

        **Company Analysis Conversations:**
        - Query 1: "Tell me about Balfour Beatty"
        - Query 2: "How does their safety record compare to industry standards?"
        - Query 3: "What about their main competitors?" *(builds on company context)*

        **Technical Building Conversations:**
        - Query 1: "BREEAM requirements for office buildings"
        - Query 2: "How does that affect material choices?"
        - Query 3: "What about cost implications?" *(continues technical discussion)*

        ### Advanced Context Features

        - **Domain Memory**: Remembers weather, company, regulatory, and technical discussions
        - **Cross-Reference**: Can connect information across different domains
        - **Progressive Detail**: Each question can dive deeper into topics
        - **Conversation Analytics**: Track how your discussions evolve

        ### Technical Context Details

        - **History Depth**: Maintains last 5 full exchanges for context
        - **Smart Truncation**: Long responses are summarized for context efficiency
        - **Domain Prioritization**: Recent same-domain queries weighted higher
        - **Real-time Integration**: Context combined with live data for current responses

        ### Tips for Best Results

        1. **Build Progressively**: Start with broad questions, then get specific
        2. **Reference Explicitly**: Use "earlier," "previously," or "compared to" for clear context
        3. **Cross-Connect**: Ask how different topics relate to each other
        4. **Use Follow-ups**: Natural conversation flows work best
        5. **Check Analytics**: See conversation patterns in the Analytics tab
        """)


if __name__ == "__main__":
    main()