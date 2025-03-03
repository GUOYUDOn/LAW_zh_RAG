from elasticsearch import Elasticsearch

es = Elasticsearch(
    hosts=["https://localhost:9200"],
    basic_auth=("elastic", "your passwords"),  
    ca_certs="your certs",                  
    verify_certs=True
)

if __name__ == "__main__":
    if es.ping():
        print("Connected to Elasticsearch!")
    else:
        print("Connection failed.")
