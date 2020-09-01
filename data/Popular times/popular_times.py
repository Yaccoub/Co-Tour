#!/usr/bin/env python
# coding: utf-8

# In[58]:


import urllib
import urllib.request
import json
import ssl


# functions
def write_to_file(file, text):
    text += '\n'
    file.write(text)


def sanitize(word):
    if type(word) != 'str':
        return str(word)
    else:
        return str(''.join([x for x in str(word) if ord(x) < 128]))


# dictionaries
days = {7: "Sunday", 1: "Monday", 2: "Tuesday", 3: "Wednesday", 4: "Thursday", 5: "Friday", 6: "Saturday"}
hours = {0: '12:00 AM', 1: '1:00 AM', 2: '2:00 AM', 3: '3:00 AM', 4: '4:00 AM', 5: '5:00 AM', 6: '6:00 AM',
         7: '7:00 AM', 8: '8:00 AM', 9: '9:00 AM', 10: '10:00 AM', 11: '11:00 AM', 12: '12:00 PM', 13: '1:00 PM',
         14: '2:00 PM', 15: '3:00 PM', 16: '4:00 PM', 17: '5:00 PM', 18: '6:00 PM', 19: '7:00 PM', 20: '8:00 PM',
         21: '9:00 PM', 22: '10:00 PM', 23: '11:00 PM'}

myssl = ssl.create_default_context();
myssl.check_hostname = False
myssl.verify_mode = ssl.CERT_NONE

query = 'Muenchner_Philharmoniker'
query = query.replace(" ", "+")
query = urllib.parse.quote_plus(query)
url = 'https://www.google.com/search?tbm=map&fp=1&authuser=0&hl=en&pb=!4m9!1m3!1d1389.8469354983167!2d-69.7606!3d44.3301504!2m0!3m2!1i1366!2i187!4f13.1!7i20!10b1!12m6!2m3!5m1!6e2!20e3!10b1!16b1!19m3!2m2!1i392!2i106!20m48!2m2!1i203!2i100!3m1!2i4!6m6!1m2!1i86!2i86!1m2!1i408!2i256!7m34!1m3!1e1!2b0!3e3!1m3!1e2!2b1!3e2!1m3!1e2!2b0!3e3!1m3!1e3!2b0!3e3!1m3!1e4!2b0!3e3!1m3!1e8!2b0!3e3!1m3!1e3!2b1!3e2!1m3!1e9!2b1!3e2!2b1!4b1!9b0!22m6!1sJ9tJWKzNFebe0gLknaaACQ%3A11!2zMWk6MCx0OjExODg2LGU6MCxwOko5dEpXS3pORmViZTBnTGtuYWFBQ1E6MTE!7e81!12e5!17sJ9tJWKzNFebe0gLknaaACQ%3A21!18e15!24m12!2b1!5m3!2b1!5b1!6b1!10m1!8e3!17b1!24b1!25b1!30m1!2b1!26m3!2m2!1i80!2i92!30m0!37m1!1e81!42b1!47m0!49m1!3b1&q=' + query + '&oq=' + query + '&gs_l=maps.3..38l5.8635.20583.1.20786.25.19.0.0.0.0.280.2398.2j7j5.14.0....0...1ac.1.64.maps..12.13.2117...38i72k1.&tch=1&ech=1&psi=J9tJWKzNFebe0gLknaaACQ.1481235308841.1'
req = urllib.request.Request(url)
req.add_header('Referer', 'https://www.google.com/')
req.add_header('User-Agent',
               'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.99 Safari/537.36')
res = urllib.request.urlopen(req, context=myssl).read().decode()
res = res.strip('/*""*/"')
dict_of_google_response = dict(json.loads(res))
long_response_value = dict_of_google_response['d'].lstrip(")]}\'\n")
j_decoded_list = json.loads(long_response_value)
l = j_decoded_list

file = open(query + ".txt", "a")
number_of_results = len(l[0][1])
for result_count in range(number_of_results):
    try:
        name = l[0][1][result_count][14][72][0][0][6][1]
        address = l[0][1][result_count][14][72][0][0][17][0]
        popular_times = l[0][1][result_count][14][84][0]
        text = "\n%s\n%s\n" % (name, address)
        print(text)
        i = 0
        for i in range(len(popular_times)):
            popular_times_day_array = popular_times[i]
            day_integer = popular_times_day_array[0]
            hour_popular_times_array = popular_times_day_array[1]
            if hour_popular_times_array != None:
                for popular_times_hour in hour_popular_times_array:
                    hour_integer = popular_times_hour[0]
                    percentage_as_hour = popular_times_hour[1]
                    comment_per_hour = popular_times_hour[2]
                    day = days[day_integer]
                    hour = hours[hour_integer]
                    text = '| {:^9} | {:^10} | {:>3} |'.format(day, hour, percentage_as_hour) + ' ' + comment_per_hour
                    print(text)
                    write_to_file(file, text)
            else:
                text = '| {:^9} |'.format(days[day_integer]) + ' closed'
                print(text)
                write_to_file(file, text)
    except TypeError as e:

        text = "\n%s\n%s\n" % (name, address)
        print(text)
        write_to_file(file, text)
        print("There are no popular times listed for this location")
        write_to_file(file, "There are no popular times listed for this location")
    except:
        pass
file.close()

# In[ ]:


# In[ ]:
