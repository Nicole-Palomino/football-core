import requests
import pandas as pd

from io import StringIO

# Función para descargar archivo desde URL
def download_csv_from_url(url, base_url=None):
    try:
        # Si la URL es relativa, construir URL completa
        if base_url and not url.startswith(('http://', 'https://')):
            url = urljoin(base_url, url)
        
        # Realizar petición
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        # Intentar leer como CSV
        csv_content = StringIO(response.text)
        df = pd.read_csv(csv_content)
        return df, None
    except Exception as e:
        return None, str(e)