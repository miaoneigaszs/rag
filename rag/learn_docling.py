try:
    import shelve
    __SHELVE_AVAILABLE = True
except ImportError:
    __SHELVE_AVAILABLE = False

if __SHELVE_AVAILABLE:
    with shelve.open('my_db', 'c') as db:
        db['name'] = 'zhangsan'
        db['age'] = 20
else:
    print("shelve not available")

