CC = gcc
#CFLAGS = -Wall -O3 -I../inc
#CFLAGS = -std=gnu99 -Wall -O3 -fipa-cp-clone -I../inc
#CFLAGS = -std=c99 -pedantic -Wall -g -I../inc
CFLAGS = -std=c99 -pedantic -Wall -O3 -I../inc

TARGET = main
OBJECTS = main.o

$(TARGET): $(OBJECTS)
	$(CC) $(CFLAGS) $(OBJECTS) -o $@ -L../lib -lzfp -lm

.c.o:
	$(CC) $(CFLAGS) -c $<

clean:
	rm -f $(TARGET) $(OBJECTS)
