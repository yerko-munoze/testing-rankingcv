import ssl
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from PyPDF2 import PdfReader

# Fix SSL Error for downloading NLTK resources
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Initialize lemmatizer and set of stopwords in Spanish
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('spanish'))

def preprocess_text(text):
    """
    Preprocess text by converting to lowercase, removing non-alphabetic characters,
    tokenizing, lemmatizing, and removing stopwords.
    """
    # Convert text to lowercase
    text = text.lower()
    # Remove non-alphabetic characters (retain letters and spaces)
    text = re.sub(r'[^a-záéíóúñü\s]', '', text)
    # Tokenize the text into words
    words = nltk.word_tokenize(text)
    # Lemmatize words and remove stopwords
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    # Join the processed words back into a single string
    return ' '.join(lemmatized_words)

def extract_text_from_pdf(pdf_file):
    """
    Extract text content from a PDF file.
    """
    try:
        # Create a PDF reader object
        pdf_reader = PdfReader(pdf_file)
        text = ""
        # Iterate through each page in the PDF
        for page in pdf_reader.pages:
            # Extract text from the page
            page_text = page.extract_text()
            if page_text:
                text += page_text + " "
        return text
    except Exception as e:
        print(f"Error reading PDF file: {str(e)}")
        return ""
