#ifndef SRC_TEXTDISPLAY_H
#define SRC_TEXTDISPLAY_H

#include <math.h>
#include <stdlib.h>
#include <stdio.h>

#include "geometry.h"
#include "macros.h"

#ifndef IMAGE_SIZE
#define IMAGE_SIZE 512
#endif

inline int sign(int x) {
    if (x > 0) return (+1);
    if (x < 0) return (-1);
    return (0);
}

struct textimage {
    char pxls[IMAGE_SIZE*IMAGE_SIZE];
};

inline void textimage_put(textimage &img, int x, int y, char v) {
    if (x >= 0 && y >= 0 && x < IMAGE_SIZE && y < IMAGE_SIZE) {
        if (x != x || y != y)
            printf("RARE NaN value detected!\n");
        else
            img.pxls[y*IMAGE_SIZE+x] = v;
    }
}

inline int round_coord(coord a) {
    return (int)(round(a*IMAGE_SIZE/DOMAIN_BOUND));
}

inline void init_textimage(textimage &img) {
    memset(img.pxls,' ',sizeof(char)*IMAGE_SIZE*IMAGE_SIZE);
}

inline void textimage_draw_simple_line(textimage &img,
    vector2 start, vector2 end, char v, bool inverted) {

    vector2 delta = vector2_delta_to(end,start);
    
    if (inverted)
        start = vector2_sum(end,vector2_rotate180(delta));
    
    int startx = round_coord(start.x);
    int starty = round_coord(start.y);
    int deltax = round_coord(delta.x);
    int deltay = round_coord(delta.y);
    
    if (deltax == 0 && deltay == 0) {
        textimage_put(img, startx, starty, v);
    } 
    else if (abs(deltax) > abs(deltay)) {
        for (int xx = 0; xx != deltax + sign(deltax); xx += sign(deltax)) {
            if(xx >= IMAGE_SIZE || xx <= -IMAGE_SIZE) break;
            int yy = (int)round((double)xx*deltay/deltax);
            textimage_put(img, startx + xx, starty + yy, v);
        }
    }
    else {
        for (int yy = 0; yy != deltay + sign(deltay); yy += sign(deltay)) {
            if (yy >= IMAGE_SIZE || yy <= -IMAGE_SIZE) break;
            int xx = (int)round((double)yy*deltax/deltay);
            textimage_put(img, startx + xx, starty + yy, v);
        }
    }
}

inline void textimage_draw_line(textimage &img, const vector2 a, const vector2 b, char v) {
    textimage_draw_simple_line(img, a, b, v, 0);
    textimage_draw_simple_line(img, a, b, v, 1);
}

inline bool textimage_to_file(const textimage &img, const char *fname) {
    FILE *fil = fopen(fname, "w");
    
    if (fil == NULL)
        return false;
    
    for (int y = 0; y < IMAGE_SIZE; y++) {
        fwrite(img.pxls + (IMAGE_SIZE * y), sizeof(char), IMAGE_SIZE, fil);
        fputc('\n',fil);
    }

    fclose(fil);
    return true;
}

inline void textimage_from_frontiers(textimage &img, boundary *boundaries, int flen) {
    init_textimage(img);

    for (int k = 0; k < flen; k++) {
        #ifndef IMGDEBUG
        if (boundaries[k].enabled) {
        #endif
            // Draw mid points:
            for (int i = 0; i < INNER_POINTS; i++) {
                int x = round_coord(boundaries[k].inners[i].x);
                int y = round_coord(boundaries[k].inners[i].y);
                for (int xx = -1; xx < 2; xx++) {
                    for (int yy = -1; yy < 2; yy++)
                        textimage_put(img, x + xx, y + yy, 'P');
                }
            }
        #ifndef IMGDEBUG
        }
        #endif
    }

    for (int k = 0; k < flen; k++) {
        #ifndef IMGDEBUG
        if (boundaries[k].enabled) {
        #endif
            char color='*';
            if (boundaries[k].enabled) color='@';
            if (boundaries[k].to_flip) color='!';
            // Draw lines
            #if INNER_POINTS == 0
                textimage_draw_line(img, boundaries[k].ini->pos, boundaries[k].end->pos, color);
            #else
                textimage_draw_line(img, boundaries[k].ini->pos, boundaries[k].inners[0], color);
                for (int i = 0; i < INNER_POINTS-1; i++) {
                    textimage_draw_line(img, boundaries[k].inners[i], boundaries[k].inners[i+1], color);
                }
                textimage_draw_line(img, boundaries[k].inners[INNER_POINTS-1], boundaries[k].end->pos, color);
            #endif
        #ifndef IMGDEBUG
        }
        #endif
    }
}

inline void textimage_to_bitmap(const textimage timg, const char* fname) {
    // Code modified from http://stackoverflow.com/questions/2654480/writing-bmp-image-in-pure-c-c-without-other-libraries#2654860
    int w = IMAGE_SIZE;
    int h = IMAGE_SIZE;
    FILE *f;
    int filesize = 54 + 3*w*h;
    unsigned char *img = (unsigned char *)malloc(3*w*h);
    memset(img, 0, sizeof(img));

    for (int i = 0; i < w; i++) {
        for (int j = 0; j < h; j++) {
            int x = i;
            int y = (h-1) - j;
            unsigned char r,g,b;

            switch (timg.pxls[j*IMAGE_SIZE+i]) {
                case ' ':
                    r = 255; g = 255; b = 255;
                    break;
                case 'P':
                    r = 255; g = 0; b = 0;
                    break;
                case '!':
                    r = 255; g = 0; b = 255;
                    break;
                case '*':
                    r = 255; g = 127; b = 255;
                    break;
                default:
                    r = 0;g = 0; b = 0;
                    break;
            }

            img[(x+y*w)*3+2] = r;
            img[(x+y*w)*3+1] = g;
            img[(x+y*w)*3+0] = b;
        }
    }

    unsigned char bmpfileheader[14] = {'B','M', 0,0,0,0, 0,0, 0,0, 54,0,0,0};
    unsigned char bmpinfoheader[40] = {40,0,0,0, 0,0,0,0, 0,0,0,0, 1,0, 24,0};
    unsigned char bmppad[3] = {0,0,0};

    bmpfileheader[ 2] = (unsigned char)(filesize    );
    bmpfileheader[ 3] = (unsigned char)(filesize>> 8);
    bmpfileheader[ 4] = (unsigned char)(filesize>>16);
    bmpfileheader[ 5] = (unsigned char)(filesize>>24);

    bmpinfoheader[ 4] = (unsigned char)(w    );
    bmpinfoheader[ 5] = (unsigned char)(w>> 8);
    bmpinfoheader[ 6] = (unsigned char)(w>>16);
    bmpinfoheader[ 7] = (unsigned char)(w>>24);
    bmpinfoheader[ 8] = (unsigned char)(h    );
    bmpinfoheader[ 9] = (unsigned char)(h>> 8);
    bmpinfoheader[10] = (unsigned char)(h>>16);
    bmpinfoheader[11] = (unsigned char)(h>>24);

    f = fopen(fname, "wb");
    fwrite(bmpfileheader, 1, 14, f);
    fwrite(bmpinfoheader, 1, 40, f);
    for (int i = 0; i < h; i++) {
        fwrite(img + (w*(h-i-1)*3), 3, w, f);
        fwrite(bmppad, 1, (4-(w*3)%4)%4, f);
    }
    fclose(f);
    free(img);
}
#endif
