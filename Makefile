all: httpd client
LIBS = -lpthread #-lsocket
httpd: httpd.c
	gcc -g -W -Wall $(LIBS) -o $@ $< -lpthread

client: simpleclient.c
	gcc -W -Wall -o $@ $<  -lpthread
clean:
	rm httpd
