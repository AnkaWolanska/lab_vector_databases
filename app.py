from sqlalchemy.engine import URL
from pgvector.sqlalchemy import Vector
from sqlalchemy import Integer, String, Column
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy import create_engine, Engine, select
from sqlalchemy.orm import Session
import numpy as np
from typing import List



db_url = URL.create(
    drivername="postgresql+psycopg",
    username="postgres",
    password="password",
    host="localhost",
    port=5555,  
    database="similarity_search_service_db"  
)


# Create the base class for the table definition
class Base(DeclarativeBase):
    __abstract__ = True


# Create the table definition
class Image(Base):
    __tablename__ = "images"
    VECTOR_LENGTH = 512
    
    # primary key
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    # image path - we will use it to store the path to the image file, after similarity search we can use it to retrieve the image and display it
    image_path: Mapped[str] = mapped_column(String(256))
    # image embedding - we will store the image embedding in this column, the image embedding is a list of 512 floats this is the output of the sentence transformer model
    image_embedding: Mapped[List[float]] = mapped_column(Vector(VECTOR_LENGTH))



# reusable function to insert data into the table
def insert_image(engine: Engine, image_path: str, image_embedding: list[float]):
    with Session(engine) as session:
        # create the image object
        image = Image(
            image_path=image_path,
            image_embedding=image_embedding
        )
        # add the image object to the session
        session.add(image)
        # commit the transaction
        session.commit()

# insert some data into the table
def insert_random_images(engine: Engine):
    N = 100
    for i in range(N):
        image_path = f"image_{i}.jpg"
        print(f"inserting { image_path }")
        image_embedding = np.random.rand(512).tolist()
        insert_image(engine, image_path, image_embedding)




# calculate the cosine similarity between the first image and the K rest of the images, order the images by the similarity score
def find_k_images(engine: Engine, k: int, orginal_image: Image) -> list[Image]:
    with Session(engine) as session:
        # execution_options={"prebuffer_rows": True} is used to prebuffer the rows, this is useful when we want to fetch the rows in chunks and return them after session is closed
        result = session.scalars(
            select(Image)
            .order_by(Image.image_embedding.cosine_distance(orginal_image.image_embedding))
            .limit(k), 
            execution_options={"prebuffer_rows": True}
        )

        return result.all()





engine = create_engine(db_url)

# create the db tables
#Base.metadata.create_all(engine)
# insert random images
#insert_random_images(engine)

# select first image from the table
with Session(engine) as session:
    image = session.query(Image).first()
    print(f"first image is {image.image_path}")

# find the 10 most similar images to the first image
k = 10
similar_images = find_k_images(engine, k, image)
for similar_image in similar_images:
    print(f" - {similar_image.image_path}")