from .database import Database

db = Database("de")
print(db.get_concept("Rheinland Pfalz")["name"])
print(db.get_shortest_path("Rheinland Pfalz", "Hogwarts"))
