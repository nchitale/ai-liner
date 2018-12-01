import scrapy

class FdaSpider(scrapy.Spider):
    name = "fda_spider"
    start_urls = ['https://www.fda.gov/MedicalDevices/ProductsandMedicalProcedures/DeviceApprovalsandClearances/Recently-ApprovedDevices/ucm558142.htm']

    def parse(self, response):
    	DATE_SELECTOR = '.footable-last-column '
    	for date in response.css(DATE_SELECTOR):
    		yield (
    			date.extract()
    		)