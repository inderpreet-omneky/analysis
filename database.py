from orator import DatabaseManager
config = {
    'mysql': {
        'driver': 'mysql',
        'host': 'omneky-dev-instance-1.cxjb8ndxveps.ap-south-1.rds.amazonaws.com',
        'database': 'rewrite',
        'user': 'admin',
        'password': 'W2fen5fWp81Zrf4jqhHU',
        'prefix': '',
        'charset':'utf8mb4'
    }
}

# from orator import DatabaseManager
# config = {
#     'mysql': {
#         'driver': 'mysql',
#         'host': 'omneky-emergency-22.mysql.database.azure.com',
#         'database': 'wave',
#         'user': 'roothikariwave@omneky-emergency-22',
#         'password': 'moka_#!pota*$#@966',
#         'prefix': '',
#         'charset':'utf8mb4'
#     }
# }
db = DatabaseManager(config)