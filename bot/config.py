max_file_size = 3.7 * (2**20) # bytes (3.7 MB)
mime_types = ['image/jpeg', 'image/pjpeg', 'image/png']

examples_content = {
    'Moscow-city': 'AgACAgIAAxkBAAMOZbO27XGq9osnzTAbQ4G5upTy2WUAAuPbMRvY0KFJc_zh9frT32MBAAMCAAN5AAM0BA',
    'River': 'AgACAgIAAxkBAAMSZbO3GD3eMQOeEgp24_8VkmNLIBcAAs_RMRsxwaFJAAEb28wd6oRbAQADAgADeQADNAQ',
    'Town by the river': 'AgACAgIAAxkBAAIDX2WzxwGSPKVWOKICPfzFtorywA8HAAJa3DEb2NChSQF3A1segFFHAQADAgADeQADNAQ'
}

examples_style = {
    'Aivazovsky': 'AgACAgIAAxkBAAMUZbO3J_spYhDeBTRd_bfQaFz_8qoAAtHRMRsxwaFJdmCHcp7NYxUBAAMCAAN5AAM0BA',
    'Picasso': 'AgACAgIAAxkBAAMYZbO3SmZjSTC2k_dj6TqwhjXMK3sAAtXRMRsxwaFJeXBSlj3T7nEBAAMCAAN5AAM0BA',
    'Random bullshit': 'AgACAgIAAxkBAAIDY2Wzxx9MOvEkgIU379umTTuF-oVUAAJb3DEb2NChSZqahhkHgFy7AQADAgADeQADNAQ'
}

content_bytes = [
    open('./examples/content/moscow_city.jpg', 'rb').read(),
    open('./examples/content/river.jpg', 'rb').read(),
    open('./examples/content/river_town.jpg', 'rb').read()
]

style_bytes = [
    open('./examples/style/aivazovsky.jpg', 'rb').read(),
    open('./examples/style/picasso.jpg', 'rb').read(),
    open('./examples/style/hz_che_eto.jpg', 'rb').read()
]
