#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 12:30:39 2023

Copyright 2023 Douglas Bowman

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""

import sqlite3
import pandas as pd

# SQLite databases are just a file
sqlite_chinook_db_filepath = "../../In-Class Exercises/Data/Chinook Database/Chinook_Sqlite.sqlite"

# Creating a connection is really like opening the file
chinook_connection = sqlite3.connect(sqlite_chinook_db_filepath)

# Can make pandas dataframes from tables
# read_sql() is a wrapper for read_sql_query, so can use either
# There is also a read_sql_table() for alchemy connection objects.
sql_data_df = pd.read_sql("SELECT * FROM album", chinook_connection)
sql_data_df = pd.read_sql_query("SELECT * FROM employee", chinook_connection)
sql_data_df = pd.read_sql_query("SELECT * FROM customer", chinook_connection)

# To get information about the database itself, we need to query 
# main.sqlite_master for a SQLite database (other databases will
# have a different table name housing this information).
sql_data_df = pd.read_sql("SELECT type,name,sql,tbl_name FROM main.sqlite_master;", chinook_connection)

# # Can make pandas dataframes from tables
sql_data_df = pd.read_sql_query("SELECT * FROM customer_support_reps", chinook_connection)

# Can create custom query
customer_support_reps_query = """
SELECT
 	Customer.CustomerID,
 	Customer.FirstName || Customer.LastName AS Customer_Name,
 	Customer.Company,
 	employee.FirstName || employee.LastName AS Support_Employee_Name,
 	employee.Title AS Employee_Title
FROM
 	Customer
LEFT JOIN
(
 	SELECT
		EmployeeID,
		FirstName,
		LastName,
		Title
 	FROM
		Employee
) employee
ON
 	Customer.SupportRepID = Employee.EmployeeID
"""
sql_data_df = pd.read_sql_query(customer_support_reps_query, chinook_connection)
print(sql_data_df)