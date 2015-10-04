""" Example PySpark ALS application
"""
from pyspark import SparkContext  # pylint: disable=import-error
from pyspark.mllib.recommendation import ALS, Rating  # pylint: disable=import-error


def parse_rating(line):
    """ Parse Movielens Rating line to Rating object.
        UserID::MovieID::Rating::Timestamp
    """
    line = line.split('::')
    return Rating(int(line[0]), int(line[1]), float(line[2]))


def parse_movie(line):
    """ Parse Movielens Movie line to Movie tuple.
        MovieID::Title::Genres
    """
    line = line.split('::')
    return (line[0], line[1])


def main():
    """ Train and evaluate an ALS recommender.
    """
    # Set up environment
    sc = SparkContext("local[*]", "RecSys")

    # Load and parse the data
    data = sc.textFile("./data/ratings.dat")
    ratings = data.map(parse_rating)

    # Build the recommendation model using Alternating Least Squares
    rank = 10
    iterations = 20
    model = ALS.train(ratings, rank, iterations)

    movies = sc.textFile("./data/movies.dat")\
               .map(parse_movie)
    # Evaluate the model on training data
    testdata = ratings.map(lambda p: (p[0], p[1]))
    predictions = model.predictAll(testdata)\
                       .map(lambda r: ((r[0], r[1]), r[2]))
    rates_and_preds = ratings.map(lambda r: ((r[0], r[1]), r[2]))\
                             .join(predictions)
    MSE = rates_and_preds.map(lambda r: (r[1][0] - r[1][1])**2).mean()
    print("Mean Squared Error = " + str(MSE))


if __name__ == "__main__":
    main()
