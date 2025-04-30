def handler(event):
    try:
        input_data = event.get("input", {})
        prompt = input_data.get("prompt")
        if not prompt:
            return {"error": "Prompt is required."}
        # 이미지 생성 로직
        result_image = generate_image(prompt)
        return {"status": "success", "image": result_image}
    except Exception as e:
        return {"error": str(e)}
