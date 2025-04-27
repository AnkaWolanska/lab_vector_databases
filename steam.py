from sqlalchemy.engine import URL
from pgvector.sqlalchemy import Vector
from sqlalchemy import Integer, String, Column, Float, Boolean
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy import create_engine, Engine, select
from sqlalchemy.orm import Session
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import numpy as np
from typing import List, Optional



db_url = URL.create(
    drivername="postgresql+psycopg",
    username="postgres",
    password="password",
    host="localhost",
    port=5555,  
    database="similarity_search_service_db"  
)

class Base(DeclarativeBase):
    __abstract__ = True

class Games(Base):
    __tablename__ = "games"
    __table_args__ = {'extend_existing': True}
    
    VECTOR_LENGTH = 512
        
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(256))
    description: Mapped[str] = mapped_column(String(4096))
    windows: Mapped[bool] = mapped_column(Boolean)
    linux: Mapped[bool] = mapped_column(Boolean)
    mac: Mapped[bool] = mapped_column(Boolean)
    price: Mapped[float] = mapped_column(Float)
    game_description_embedding: Mapped[Vector] = mapped_column(Vector(VECTOR_LENGTH))


engine = create_engine(db_url)
# Base.metadata.create_all(engine)


checkpoint = "distiluse-base-multilingual-cased-v2"
model = SentenceTransformer(checkpoint)

def generate_embeddings(text: str) -> list[float]:
    return model.encode(text)

def insert_games(engine, dataset):
    with tqdm(total=len(dataset), desc="inserting games") as pbar:
        for i, game in enumerate(dataset):
            name, windows, linux, mac, price = game["Name"], game["Windows"], game["Linux"], game["Mac"], game["Price"]
            game_description = game["About the game"] or ""
            if name and windows and linux and mac and price and game_description:
                game_embedding = generate_embeddings(game_description)
                game = Games(
                    name=game["Name"], 
                    description=game_description[0:4096],
                    windows=game["Windows"], 
                    linux=game["Linux"], 
                    mac=game["Mac"], 
                    price=game["Price"], 
                    game_description_embedding=game_embedding
                )
                with Session(engine) as session:
                    session.add(game)
                    session.commit()
            pbar.update(1)


# print("--- begin insert ---")

# N = 40000
# dataset = load_dataset("FronkonGames/steam-games-dataset")
# columns_to_keep = ["Name", "Windows", "Linux", "Mac", "About the game", "Supported languages", "Price"]
# dataset = dataset["train"].select_columns(columns_to_keep).select(range(N))

# insert_games(engine, dataset)

# print("--- end insert ---")



def find_game(
    engine: Engine, 
    game_description: str, 
    windows: Optional[bool] = None, 
    linux: Optional[bool] = None,
    mac: Optional[bool] = None,
    price: Optional[int] = None
):
    with Session(engine) as session:
        game_embedding = generate_embeddings(game_description)
    
        query = (
            select(Games)
            .order_by(Games.game_description_embedding.cosine_distance(game_embedding))
        )
        
        if price:
            query = query.filter(Games.price <= price)
        if windows:
            query = query.filter(Games.windows == True)
        if linux:
            query = query.filter(Games.linux == True)
        if mac:
            query = query.filter(Games.mac == True)
        
        result = session.execute(query, execution_options={"prebuffer_rows": True})
        game = result.scalars().first()
        
        return game
    



print("--- begin query ---")

game = find_game(engine, "This is a game about a hero who saves the world", price=10)
print(f"Game: {game.name}")
print(f"Description: {game.description}")


game = find_game(engine, game_description="Home decorating", price=20)
print(f"Game: {game.name}")
print(f"Description: {game.description}")

game = find_game(engine, game_description="Home decorating", mac=True, price=5)
print(f"Game: {game.name}")
print(f"Description: {game.description}")

print("--- end query ---")