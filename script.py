import bs4
import requests

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import GPT4AllEmbeddings

base_url = "https://friends.fandom.com"

site_map_paths = [
    "/wiki/Local_Sitemap",
    "/wiki/Local_Sitemap?namefrom=Elliott+Gould",
    "/wiki/Local_Sitemap?namefrom=Kim+Harris",
    "/wiki/Local_Sitemap?namefrom=Richard+Darnville",
    "/wiki/Local_Sitemap?namefrom=The+One+With+Ross%27+Library+Book",
]

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=500, chunk_overlap=100
)

vectorstore = Chroma(
    embedding_function=GPT4AllEmbeddings(model_name="nomic-embed-text-v1.5.f16.gguf"),
    persist_directory="./chroma_db",
    collection_name="wiki",
)


def get_page_categories(soup):
    categories = soup.find_all("a", title=lambda x: x and x.startswith("Category:"))
    return list(set(category.text.lower() for category in categories))


def scape_page(url, title):
    site = requests.get(url)
    soup = bs4.BeautifulSoup(site.content, features="html.parser")
    categories = get_page_categories(soup)

    if "joey" in categories:  # ignore joey tv show
        return

    loader = WebBaseLoader(
        url,
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                class_=("mw-page-title-main", "mw-parser-output")
            )
        ),
    )
    docs = loader.load()
    for doc in docs:
        doc.metadata["title"] = title
        # doc.metadata["tags"] = categories

        with open(f"./wiki/{title.replace('/', '_')}.txt", "w") as f:
            f.write(doc.page_content)

    return docs


def scrape_site_map(url):
    site = requests.get(url)
    soup = bs4.BeautifulSoup(site.content, features="html.parser")
    anchor_tags = soup.find("div", class_="mw-allpages-body").find_all("a")
    for anchor_tag in anchor_tags:
        title = anchor_tag.text.lower()
        if "admin" in title:  # skip administrator page
            continue

        print("title:", title)

        page_docs = scape_page(url=base_url + anchor_tag["href"], title=title)

        if page_docs is None:
            continue
        splits = text_splitter.split_documents(page_docs)
        vectorstore.add_documents(splits)


def main():
    for site_map_path in site_map_paths:
        scrape_site_map(base_url + site_map_path)


if __name__ == "__main__":
    main()
