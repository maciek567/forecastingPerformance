from run.configuration import company_names, create_stock

for company_name in company_names:
    stock = create_stock(company_name, is_cache=True)

print("DONE")
