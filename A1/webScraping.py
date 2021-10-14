from selenium import webdriver as wd
from bs4 import BeautifulSoup as btsp
import pandas

def web_scraper(site, state, pages):
	dr = wd.Chrome("/usr/lib/chromium-browser/chromedriver")
	hotel_name, city, country, star_rating, price = [], [], [], [], []
	facility, about, reviews = [], [], []

	for p in range(pages):
		dr.get(site+"/hotels/"+state+"?page="+str(p+1))
		soup = btsp(dr.page_source, 'lxml')

		section = soup.findAll('div', attrs={'class':'clearfix p8 relative wfull min-width-0'})
		for sec in section:
			#hotel name
			hotel = sec.find('a', attrs={'class':'at_hotelName f14 m0 mr8 fw9 text-capitalize _3Pdyrtu'})
			hotel_name.append(hotel.text)
			#city
			location = sec.find('div', attrs={'class':'m0 f12'})
			s = location.text
			itr = s.find('|')
			city.append(s[:itr])
			#country
			country.append(state)
			#star rating
			rt = sec.find('span', attrs={'class':'f14 fw9 sfcw'})
			star_rating.append(float(rt.text))
			#price
			pay = sec.find('p', attrs={'class':'f16 fw7 m0 sfc3'})
			if pay != None:
				pay = pay.text
				j = pay.find('-')
				if j == -1:
					price.append(pay)
				else:
					low = pay[:j].strip()
					j = pay.find('\n')
					high = pay[(j+1):].strip()
					price.append(low+" - "+high)
			else:
				price.append('NA')
			#facility
			ref = hotel.attrs['href']
			dr.get(site+ref)
			hotel_soup = btsp(dr.page_source, 'lxml')

			fac = hotel_soup.find('div', attrs={'name':'facilities-sec', 'class':'container mb48'})
			if fac != None:
				feat = fac.findAll('span', attrs={'class':'inclusionName'})
				arr = []
				for f in feat:
					arr.append(f.text)
				facility.append(arr)
			else:
				facility.append([])
			#about
			description = hotel_soup.find('div', attrs={'name':'overview-sec', 'class':'clearfix mb48'})
			if description != None:
				matter = description.find('p')
				about.append(matter.text)
			else:
				about.append('Hotel Sold Out');
			#reviews
			review = hotel_soup.find('div', attrs={'name':'testimonial-sec', 'class':'clearfix pt24 pb24 sbc5 mb48'})
			if review != None:
				rev = review.find('div', attrs={'class':'dynamicTextInherit f14p mt15 mb24'})
				rev = rev.find('p')
				reviews.append(rev.text)
			else:
				reviews.append('Not Available')
	## converting to CSV
	excel = pandas.DataFrame({'Hotel Name':hotel_name, 'City':city, 'Country':country, 'Star Rating':star_rating, 'Price per night':price, 'Amenities (array)':facility, 'Hotel Description':about, 'Review':reviews})
	excel.to_csv('Rajat.csv', index=False, encoding='utf-8')


site = input('Enter site name :')
state = input('Enter state name :')
pages = input('Enter no. of pages :')

web_scraper(site, state, int(pages))
