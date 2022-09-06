import pymongo



client = pymongo.MongoClient('mongodb://localhost:27017')
print(client)
db = client['Pra']
collection = db['mySamplecollectionForPra']
# dictionary={'Name': 'Pranav', 'Marks': 50}
# collection.insert_one(dictionary)
# dictionary={'Name': 'Pranav 2', 'Marks': 45}
# collection.insert_one(dictionary)

insertThese=[
    {'Name': 'Pranav', 'Location': 'Delhi', 'Mob': 9765442249},
    {'Name': 'Praveen', 'Location': 'Delhi', 'Mob': 9765442249},
    {'Name': 'Adarsh', 'Location': 'Noida', 'Mob': 9765442788}
]
collection.insert_many(insertThese)