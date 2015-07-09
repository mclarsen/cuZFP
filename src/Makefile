CC = gcc
#CFLAGS = -std=gnu99 -Wall -O3 -msse4 -I../inc
#CFLAGS = -std=c99 -pedantic -Wall -O3 -msse4 -fverbose-asm -S -ftree-vectorizer-verbose=2 -I../inc
CFLAGS = -std=c99 -pedantic -Wall -O3 -I../inc
#CFLAGS = -std=c89 -pedantic -Wall -Wno-long-long -Wno-unused-function -O3 -I../inc
#CFLAGS = -std=c99 -pedantic -Wall -g -I../inc

TARGET = ../lib/libzfp.a
OBJECTS = bitstream.o decode1f.o decode1d.o encode1f.o encode1d.o decode2f.o decode2d.o encode2f.o encode2d.o decode3f.o decode3d.o encode3f.o encode3d.o

$(TARGET): $(OBJECTS)
	rm -f $(TARGET)
	ar rc $(TARGET) $(OBJECTS)

.c.o:
	$(CC) $(CFLAGS) -c $<

clean:
	rm -f $(TARGET) $(OBJECTS)
