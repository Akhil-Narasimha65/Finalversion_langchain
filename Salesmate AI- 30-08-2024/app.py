import os
import re
import numpy as np
from collections import defaultdict
from dotenv import load_dotenv
from pdfminer.high_level import extract_text
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from openai import OpenAIError
import time
from scipy import stats

# Load environment variables
load_dotenv()

# Set your OpenAI API key
openai_api_key = os.getenv("OPENAI_API_KEY")

# Initialize session state
if 'metrics' not in st.session_state:
    st.session_state.metrics = None
if 'pdf_processed' not in st.session_state:
    st.session_state.pdf_processed = False
if 'full_text' not in st.session_state:
    st.session_state.full_text = ""
if 'qa' not in st.session_state:
    st.session_state.qa = None

def embed_with_retry(embedding_func, *args, **kwargs):
    max_retries = 5
    retry_delay = 1  # Start with a 1-second delay

    for attempt in range(max_retries):
        try:
            return embedding_func(*args, **kwargs)
        except OpenAIError as e:
            if "RateLimitError" in str(e):
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    raise e  # Reraise the error if max retries are exceeded
            else:
                raise e  # Reraise other OpenAI errors

def process_pdf(file):
    with open("temp.pdf", "wb") as f:
        f.write(file.getbuffer())

    text = extract_text("temp.pdf")
    st.session_state.full_text = text  # Store the full text in session state
    
    # Improved text splitting strategy
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
    )
    chunks = text_splitter.split_text(text)

    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    db = embed_with_retry(FAISS.from_texts, chunks, embedding=embeddings)

    return db, text

def extract_metrics(text):
    metrics = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))

    # Updated pattern to match the new structure of your PDF content
    pattern = r'HCP\s+([A-Za-z\s]+)\s+WITH\s+NPI\s+(\d+)\s+HAS\s+([\d.]+)\s+C1W\s+NRX,\s+([\d.]+)\s+P1W\s+NRX,\s+([\d.]+)\s+C1W\s+TRX,\s+([\d.]+)\s+P1W\s+TRX,\s+([\d.]+)\s+C1W\s+NBRX,\s+([\d.]+)\s+P1W\s+NBRX,\s+([\d.]+)\s+C1W\s+DISP\s+CLAIMS,\s+([\d.]+)\s+P1W\s+DISP\s+CLAIMS'

    matches = list(re.finditer(pattern, text, re.IGNORECASE))

    for match in matches:
        hcp_name = match.group(1).strip()
        npi = match.group(2)
        
        metrics[hcp_name]['npi'] = npi
        metrics[hcp_name]['nrx']['current'] = float(match.group(3))
        metrics[hcp_name]['nrx']['previous'] = float(match.group(4))
        metrics[hcp_name]['trx']['current'] = float(match.group(5))
        metrics[hcp_name]['trx']['previous'] = float(match.group(6))
        metrics[hcp_name]['nbrx']['current'] = float(match.group(7))
        metrics[hcp_name]['nbrx']['previous'] = float(match.group(8))
        metrics[hcp_name]['disp claims']['current'] = float(match.group(9))
        metrics[hcp_name]['disp claims']['previous'] = float(match.group(10))

    # Display extracted metrics to verify correctness

    first_two_metrics = dict(list(metrics.items())[:2])
    st.write("First Two Extracted Metrics:", first_two_metrics)

    return metrics




def answer_question(question, metrics):
    lower_question = question.lower()

    # Metric aliases
    metric_aliases = {
        'trx': ['total rx', 'total prescriptions'],
        'nrx': ['new rx', 'new prescriptions'],
        'nbrx': ['new-to-brand rx', 'new to brand prescriptions'],
        'disp claims': ['dispensed claims', 'dispensing claims']
    }

    def get_metric(metric_name):
        for key, aliases in metric_aliases.items():
            if metric_name in [key] + aliases:
                return key
        return metric_name

    # Handle complex queries with multiple conditions
    if 'list all hcps' in lower_question:
        conditions = []
        for metric in ['trx', 'nrx', 'nbrx', 'disp claims']:
            above_match = re.search(fr'{metric} above (\d+(?:\.\d+)?)', lower_question)
            below_match = re.search(fr'{metric} below (\d+(?:\.\d+)?)', lower_question)
            if above_match:
                threshold = float(above_match.group(1))
                conditions.append(lambda hcp, data, m=metric, t=threshold: data[m]['current'] > t)
            if below_match:
                threshold = float(below_match.group(1))
                conditions.append(lambda hcp, data, m=metric, t=threshold: data[m]['current'] < t)
        
        matching_hcps = [
            hcp for hcp, data in metrics.items()
            if all(condition(hcp, data) for condition in conditions)
        ]
        
        return f"HCPs meeting the criteria: {', '.join(matching_hcps)}"

    # Handle top N HCPs with multiple metrics
    top_n_match = re.search(r'top (\d+) hcps', lower_question)
    if top_n_match:
        n = int(top_n_match.group(1))
        primary_metric_match = re.search(r'with high(est)? (\w+)', lower_question)
        secondary_metric_match = re.search(r'what is their (\w+)', lower_question)
        
        if primary_metric_match and secondary_metric_match:
            primary_metric = get_metric(primary_metric_match.group(2).lower())
            secondary_metric = get_metric(secondary_metric_match.group(1).lower())
            
            # Sort HCPs by primary metric
            sorted_hcps = sorted(metrics.items(), key=lambda x: x[1][primary_metric]['current'], reverse=True)[:n]
            
            result = f"The top {n} HCPs for {primary_metric.upper()} (current week) are:\n"
            for i, (hcp, data) in enumerate(sorted_hcps, 1):
                primary_value = data[primary_metric]['current']
                secondary_value = data[secondary_metric]['current']
                result += f"{i}. {hcp}: {primary_metric.upper()}={primary_value:.2f}, {secondary_metric.upper()}={secondary_value:.2f}\n"
            
            return result

    # Extract key information from the question
    metric_match = re.search(r'(trx|nrx|nbrx|disp claims|sales|revenue|market share|growth|units sold|scripts|new scripts|renewed scripts|call rate|reach|frequency)', lower_question, re.IGNORECASE)
    aggregation_match = re.search(r'(sum|total|average|mean|max|maximum|min|minimum|count)', lower_question, re.IGNORECASE)
    ranking_match = re.search(r'(highest|top|best|lowest|bottom|worst)', lower_question, re.IGNORECASE)
    all_hcps_match = re.search(r'(all\s+HCPs|all\s+doctors|all\s+physicians)', lower_question, re.IGNORECASE)
    hcp_match = re.search(r'of\s+([A-Za-z\s]+)', lower_question, re.IGNORECASE)
    period_match = re.search(r'(c1w|p1w|current|previous)', lower_question, re.IGNORECASE)
    top_n_match = re.search(r'top\s+(\d+)', lower_question, re.IGNORECASE)
    bottom_n_match = re.search(r'bottom\s+(\d+)', lower_question, re.IGNORECASE)
    compare_match = re.search(r'compared to', lower_question, re.IGNORECASE)
    trend_match = re.search(r'(change|growth|improvement)', lower_question, re.IGNORECASE)
    ratio_match = re.search(r'ratio of (\w+) to (\w+)', lower_question, re.IGNORECASE)
    percentage_match = re.search(r'percentage|percent', lower_question, re.IGNORECASE)
    distribution_match = re.search(r'(distribution|quartiles|standard deviation)', lower_question, re.IGNORECASE)

    if not metric_match:
        return "I couldn't identify a specific metric in your question. Please specify a metric such as TRX, NRX, NBRX, or DISP CLAIMS."
    
    metric = get_metric(metric_match.group(1).lower())
    aggregation = aggregation_match.group(1).lower() if aggregation_match else None
    ranking = ranking_match.group(1).lower() if ranking_match else None
    all_hcps = bool(all_hcps_match)
    hcp_name = hcp_match.group(1).strip() if hcp_match and not all_hcps else None
    period = period_match.group(1).lower() if period_match else 'both'
    top_n = int(top_n_match.group(1)) if top_n_match else None
    bottom_n = int(bottom_n_match.group(1)) if bottom_n_match else None
    compare = bool(compare_match)

    # Convert period to 'current' or 'previous'
    if period in ['c1w', 'current']:
        period = 'current'
    elif period in ['p1w', 'previous']:
        period = 'previous'

    # Handle top/bottom N HCPs question
    if top_n or bottom_n:
        n = top_n or bottom_n
        hcp_values = {}
        for hcp_name, data in metrics.items():
            if metric in data:
                if period == 'current':
                    value = data[metric]['current']
                elif period == 'previous':
                    value = data[metric]['previous']
                else:
                    value = data[metric]['current'] + data[metric]['previous']
                hcp_values[hcp_name] = value
        
        sorted_hcps = sorted(hcp_values.items(), key=lambda x: x[1], reverse=bool(top_n))
        selected_hcps = sorted_hcps[:n]
        
        result = f"The {'top' if top_n else 'bottom'} {n} HCPs for {metric.upper()} ({period} week) are:\n"
        for i, (hcp, value) in enumerate(selected_hcps, 1):
            result += f"{i}. {hcp}: {value:.2f}\n"
        return result

    # Handle comparison to average
    if compare and hcp_name:
        all_values = []
        hcp_value = None
        for name, data in metrics.items():
            if metric in data:
                if period == 'current':
                    value = data[metric]['current']
                elif period == 'previous':
                    value = data[metric]['previous']
                else:
                    value = data[metric]['current'] + data[metric]['previous']
                all_values.append(value)
                if name.lower() == hcp_name.lower():
                    hcp_value = value
        
        if hcp_value is None:
            return f"I couldn't find data for the HCP named '{hcp_name}'."
        
        average = np.mean(all_values)
        difference = hcp_value - average
        percentage = (difference / average) * 100
        
        return f"The {period} {metric.upper()} for {hcp_name} is {hcp_value:.2f}, which is {abs(difference):.2f} {'above' if difference > 0 else 'below'} the average of {average:.2f}. This is a {abs(percentage):.2f}% {'increase' if difference > 0 else 'decrease'} compared to the average."

    # Handle trend analysis
    if trend_match:
        metric_changes = {}
        for hcp, data in metrics.items():
            if metric in data:
                change = data[metric]['current'] - data[metric]['previous']
                metric_changes[hcp] = change
        
        if ranking:
            top_change = max(metric_changes.items(), key=lambda x: x[1])
            return f"The HCP with the highest {metric.upper()} improvement is {top_change[0]} with a change of {top_change[1]:.2f}."
        else:
            return f"The average change in {metric.upper()} across all HCPs is {np.mean(list(metric_changes.values())):.2f}."

    # Handle ratio analysis
    if ratio_match:
        metric1, metric2 = ratio_match.groups()
        ratios = {}
        for hcp, data in metrics.items():
            if metric1 in data and metric2 in data and data[metric2]['current'] != 0:
                ratio = data[metric1]['current'] / data[metric2]['current']
                ratios[hcp] = ratio
        
        top_ratio = max(ratios.items(), key=lambda x: x[1])
        return f"The HCP with the highest ratio of {metric1.upper()} to {metric2.upper()} is {top_ratio[0]} with a ratio of {top_ratio[1]:.2f}."

    # Handle percentage-based analysis
    if percentage_match:
        if 'total' in lower_question:
            total = sum(data[metric]['current'] for data in metrics.values() if metric in data)
            percentages = {hcp: (data[metric]['current'] / total) * 100 for hcp, data in metrics.items() if metric in data}
            top_percentage = max(percentages.items(), key=lambda x: x[1])
            return f"{top_percentage[0]} accounts for the highest percentage of total {metric.upper()} at {top_percentage[1]:.2f}%."
        elif 'above average' in lower_question:
            values = [data[metric]['current'] for data in metrics.values() if metric in data]
            avg = np.mean(values)
            above_avg = sum(1 for v in values if v > avg)
            percentage = (above_avg / len(values)) * 100
            return f"{percentage:.2f}% of HCPs have a {metric.upper()} above the average of {avg:.2f}."

    # Handle distribution analysis
    if distribution_match:
        values = [data[metric]['current'] for data in metrics.values() if metric in data]
        if 'quartiles' in lower_question:
            q1, q2, q3 = np.percentile(values, [25, 50, 75])
            return f"The quartiles for {metric.upper()} are: Q1 = {q1:.2f}, Q2 (median) = {q2:.2f}, Q3 = {q3:.2f}."
        elif 'standard deviation' in lower_question:
            std_dev = np.std(values)
            return f"The standard deviation of {metric.upper()} across all HCPs is {std_dev:.2f}."

    # Handle ranking questions (e.g., "Who has the highest total NRX?")
    if ranking:
        hcp_totals = {}
        for hcp_name, data in metrics.items():
            if metric in data:
                if period == 'current':
                    total_value = data[metric]['current']
                elif period == 'previous':
                    total_value = data[metric]['previous']
                else:
                    total_value = data[metric]['current'] + data[metric]['previous']
                hcp_totals[hcp_name] = total_value

        if not hcp_totals:
            return f"No data found for metric: {metric}."

        if ranking in ['highest', 'top', 'best']:
            ranked_hcp = max(hcp_totals, key=hcp_totals.get)
        elif ranking in ['lowest', 'bottom', 'worst']:
            ranked_hcp = min(hcp_totals, key=hcp_totals.get)
        else:
            return f"I don't understand the ranking method: {ranking}."

        total_value = hcp_totals[ranked_hcp]
        return f"The HCP with the {ranking} {metric.upper()} for {period} week is {ranked_hcp} with {total_value:.2f}."

    # Handle questions about all HCPs or specific HCPs
    if all_hcps or aggregation:
        values = []
        for hcp_data in metrics.values():
            if metric in hcp_data:
                if period == 'current':
                    values.append(hcp_data[metric]['current'])
                elif period == 'previous':
                    values.append(hcp_data[metric]['previous'])
                else:
                    values.append(hcp_data[metric]['current'] + hcp_data[metric]['previous'])
        
        if not values:
            return f"I couldn't find data for the metric: {metric}."

        result_value = handle_aggregation(values, aggregation or 'average')
        return f"The {aggregation or 'average'} {metric.upper()} across all HCPs for {period} week is {result_value:.2f}."
    elif hcp_name:
        # Case-insensitive matching for HCP names
        matching_hcps = [name for name in metrics.keys() if name.lower() == hcp_name.lower()]
        if matching_hcps:
            hcp_name = matching_hcps[0]  # Use the first match (there should only be one)
            if metric in metrics[hcp_name]:
                value = metrics[hcp_name][metric][period]
                return f"The {period} {metric.upper()} for {hcp_name} is {value:.2f}."
            else:
                return f"I couldn't find data for the metric '{metric}' for {hcp_name}."
        else:
            return f"I couldn't find data for the HCP named '{hcp_name}'."

    # If we reach here, we couldn't process the question
    return "I couldn't process your question. Please try rephrasing it or provide more specific details."

def handle_aggregation(values, aggregation):
    if not values:
        return None
    if aggregation == 'average':
        return np.mean(values)
    elif aggregation in ['sum', 'total']:
        return np.sum(values)
    elif aggregation in ['maximum', 'max']:
        return np.max(values)
    elif aggregation in ['minimum', 'min']:
        return np.min(values)
    elif aggregation == 'count':
        return len(values)
    return None


# Streamlit UI
st.title("PDF Q&A Bot")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file is not None:
    if not st.session_state.pdf_processed:
        with st.spinner("Processing PDF..."):
            try:
                db, full_text = process_pdf(uploaded_file)
                st.session_state.metrics = extract_metrics(full_text)
                st.session_state.pdf_processed = True

                AGGREGATION_PROMPT = PromptTemplate.from_template("""
                Given the following conversation and a followup question, rephrase the followup question to be a standalone question.
                If the question requires any calculations or aggregations, please perform them and show your work.
                Make sure to filter the data based on any specified conditions (e.g., region, person, division, territory, HCP, doctor).
                Provide a step-by-step breakdown of your calculations.

                Chat History: {chat_history}
                Follow up Input: {question}

                Standalone question with calculations (if needed):
                """)

                llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
                st.session_state.qa = ConversationalRetrievalChain.from_llm(
                    llm=llm,
                    retriever=db.as_retriever(),
                    condense_question_prompt=AGGREGATION_PROMPT,
                    return_source_documents=True,
                    verbose=False
                )

                st.success("PDF processed successfully!")
            except Exception as e:
                st.error(f"An error occurred while processing the PDF: {str(e)}")
    else:
        st.info("PDF has already been processed.")

    st.header("Ask a question about your PDF")
    user_question = st.text_input("Enter your question:")

    if st.button("Get Answer"):
        if user_question and st.session_state.metrics is not None:
            with st.spinner("Getting answer..."):
                try:
                    qa_result = st.session_state.qa({"question": user_question, "chat_history": []})
                    processed_answer = answer_question(user_question, st.session_state.metrics)
                    st.markdown(f"**You asked:** {user_question}")
                    st.markdown(f"**Answer:** {processed_answer}")
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
        elif st.session_state.metrics is None:
            st.error("Please upload a PDF file first.")
        else:
            st.error("Please enter a question.")
else:
    st.info("Please upload a PDF file to get started.")

# Add a button to show the full extracted text
if st.button("Show Full Extracted Text"):
    st.text_area("Full Extracted Text", st.session_state.full_text, height=300)
