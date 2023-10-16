#ifndef _YOLOX_CONFIG_H_
#define _YOLOX_CONFIG_H_

// nested classes
typedef struct GridAndStride
{
    int grid0;
    int grid1;
    int stride;
} YoloXAnchor;

typedef struct
{
    float r;
    int dw;
    int dh;
    int new_unpad_w;
    int new_unpad_h;
    bool flag;
} YoloXScaleParams;

#endif