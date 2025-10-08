#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libswscale/swscale.h>
#include <libavutil/imgutils.h>

#include "rpihub75.h"
#include "video.h"
#include "pixels.h"
#include "util.h"

/**
 * @brief pass this function to your pthread_create() call to render a video file
 * will render the video file pointed to by scene->shader_file until
 * scene->do_render is false;
 * 
 * @param arg 
 * @return void* 
 */

void* render_video_fn(void *arg) {
    scene_info *scene = (scene_info*)arg;
    while (scene->do_render) {
        if (!hub_render_video(scene, scene->shader_file)) {
            break;
        }
    }

    return NULL;
}

#define FAIL(MSG) do { fprintf(stderr, "%s\n", MSG); ok = false; goto cleanup; } while (0)



bool hub_render_video(scene_info *scene, const char *filename) {
    AVFormatContext *format_ctx = NULL;
    AVCodecContext  *codec_ctx  = NULL;
    const AVCodec *codec  = NULL;  // const per modern FFmpeg API (av_find_best_stream expects const AVCodec**)
    AVFrame *frame = NULL, *frame_rgb = NULL;
    AVPacket *packet = NULL;
    struct SwsContext *sws_ctx = NULL;
    uint8_t *rgb_tight = NULL;
    bool ok = true;

    int video_stream_index = -1;
    scene->stride = 3;

    if (avformat_open_input(&format_ctx, filename, NULL, NULL) != 0) { FAIL("Could not open video file"); }
    if (avformat_find_stream_info(format_ctx, NULL) < 0) { FAIL("Could not find stream information"); }

    // best stream selection
    codec = NULL;
    video_stream_index = av_find_best_stream(format_ctx, AVMEDIA_TYPE_VIDEO, -1, -1, &codec, 0);
    if (video_stream_index < 0 || !codec) { FAIL("No video stream found"); }

    // codec ctx
    codec_ctx = avcodec_alloc_context3((const AVCodec*)codec); // cast for older headers if needed
    if (!codec_ctx) { FAIL("Failed to allocate codec context"); }
    if (avcodec_parameters_to_context(codec_ctx, format_ctx->streams[video_stream_index]->codecpar) < 0) {
        FAIL("avcodec_parameters_to_context failed");
    }
    codec_ctx->thread_type  = FF_THREAD_FRAME;
    codec_ctx->thread_count = 2;
    if (avcodec_open2(codec_ctx, codec, NULL) < 0) { FAIL("Could not open codec"); }

    // frames and packet
    frame = av_frame_alloc();
    frame_rgb = av_frame_alloc();
    packet = av_packet_alloc();
    if (!frame || !frame_rgb || !packet) { FAIL("Could not allocate frame/packet"); }

    // tightly packed RGB24 dest
    size_t tight_row_bytes = scene->width * 3;
    rgb_tight = av_malloc((size_t)scene->height * tight_row_bytes);
    if (!rgb_tight) { FAIL("rgb_tight alloc failed"); }
    frame_rgb->data[0] = rgb_tight;
    frame_rgb->linesize[0] = (int)tight_row_bytes;

    // scaler
    int sws_flags = SWS_POINT; // or SWS_FAST_BILINEAR
    sws_ctx = sws_getContext(codec_ctx->width, codec_ctx->height, codec_ctx->pix_fmt,
                             scene->width, scene->height, AV_PIX_FMT_RGB24,
                             sws_flags, NULL, NULL, NULL);
    if (!sws_ctx) { FAIL("sws_getContext failed"); }

    // proper colorspace setup (no bogus casts)
    const int *src_mat = sws_getCoefficients(SWS_CS_ITU601);
    const int *dst_mat = sws_getCoefficients(SWS_CS_DEFAULT);
    // srcRange=0, dstRange=0 (limited), brightness=0, contrast=1.0, saturation=1.0
    if (sws_setColorspaceDetails(sws_ctx, src_mat, 0, dst_mat, 0, 0, 1<<16, 1<<16) < 0) {
        // not fatal, continue
    }

    while (av_read_frame(format_ctx, packet) >= 0) {   // NOTE: packet (not &packet)
        if (!scene->do_render) break;

        if (packet->stream_index == video_stream_index) {
            int response = avcodec_send_packet(codec_ctx, packet); // NOTE: packet (not &packet)
            if (response < 0) { FAIL("Error sending packet for decoding"); }

            while (response >= 0) {
                response = avcodec_receive_frame(codec_ctx, frame);
                if (response == AVERROR(EAGAIN) || response == AVERROR_EOF) break;
                if (response < 0) { FAIL("Error during decoding"); }

                sws_scale(sws_ctx,
                          (const uint8_t * const*)frame->data, frame->linesize,
                          0, codec_ctx->height,
                          frame_rgb->data, frame_rgb->linesize);

                map_byte_image_to_bcm(scene, frame_rgb->data[0]);

                // optional: show fps
                AVRational fr = format_ctx->streams[video_stream_index]->avg_frame_rate;
                float fps = (float)av_q2d(fr);
                calculate_fps((uint16_t)(fps > 1e-3f ? fps : 30.0f), scene->show_fps);
            }
        }
        av_packet_unref(packet); // NOTE: packet (not &packet)
    }

cleanup:
    if (packet) av_packet_free(&packet);    // expects &packet
    if (sws_ctx) sws_freeContext(sws_ctx);
    if (rgb_tight) av_free(rgb_tight);
    if (frame_rgb) av_frame_free(&frame_rgb);
    if (frame) av_frame_free(&frame);
    if (codec_ctx) avcodec_free_context(&codec_ctx);
    if (format_ctx) avformat_close_input(&format_ctx);
    return ok;
}
