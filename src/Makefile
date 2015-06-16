CC = gcc
#CFLAGS = -std=c99 -pedantic -Wall -O3 -I../inc
CFLAGS = -std=c99 -pedantic -Wall -g -I../inc

TARGET = ../lib/libzfp.a
OBJECTS = bitstream.o decode.o encode.o

$(TARGET): $(OBJECTS)
	rm -f $(TARGET)
	ar rc $(TARGET) $(OBJECTS)

.c.o:
	$(CC) $(CFLAGS) -c $<

clean:
	rm -f $(TARGET) $(OBJECTS)
