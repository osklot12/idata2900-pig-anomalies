import grpc
import asyncio
import pipeline_server_pb2
import pipeline_server_pb2_grpc

WORKER_1 = "10.22.171.164:50051"

async def send_processed_data():
    async with grpc.aio.insecure_channel(WORKER_1) as channel:
        stub = pipeline_server_pb2_grpc.PipelineServerStub(channel)
        async def request_iterator():
            for i in range(3):  # Send 3 frames
                yield pipeline_server_pb2.ProcessedFrameRequest(
                    video_name="video1",
                    frame_name=f"frame{i+1}.jpg",
                    frame_data=b"frame_bytes",
                    annotation_json='{"objects": [{"label": "object1"}]}'
                )
                await asyncio.sleep(1)  # Simulate delay

        try:
            response = await stub.SendProcessedData(request_iterator())
            print(f"✅ Server Response: {response.message}")
        except grpc.RpcError as e:
            print(f"❌ gRPC Error: {e.code()} - {e.details()}")

async def request_batches():
    async with grpc.aio.insecure_channel(WORKER_1) as channel:
        stub = pipeline_server_pb2_grpc.PipelineServerStub(channel)
        request = pipeline_server_pb2.BatchRequest(batch_size=2)
        async for batch in stub.StreamProcessedBatches(request):
            print(f"📦 Received batch with {len(batch.frames)} frames.")

async def main():
    await asyncio.gather(
        send_processed_data(),
        request_batches()
    )

if __name__ == "__main__":
    asyncio.run(main())
