import os
import json
from groq import Groq

# Default static fallback information for fault stages
STAGE_FALLBACK_INFO = {
    0: {"stage": "Stage 0 - Healthy Baseline", "crack_length": "0 mm", "description": "Turbine operates normally under baseline design parameters without detected structural cracks.", "recommendation": "Continue standard operational schedule and periodic routine inspections."},
    1: {"stage": "Stage 1 - Micro Crack Initiation", "crack_length": "15 mm", "description": "Incipient surface crack detected on trailing edge. Minor flow turbulence induced.", "recommendation": "Log vibration baseline and schedule acoustic monitoring at next planned shutdown."},
    2: {"stage": "Stage 2 - Early Crack Propagation", "crack_length": "30 mm", "description": "Crack expanding along trailing edge. Slight increase in pressure fluctuations.", "recommendation": "Increase monitoring frequency. Inspect runner blade dye penetrant / NDT during maintenance."},
    3: {"stage": "Stage 3 - Substantial Crack Growth", "crack_length": "50 mm", "description": "Crack length reached 50mm. Structural stiffness slightly reduced.", "recommendation": "Plan blade repair/welding session within upcoming operational window."},
    4: {"stage": "Stage 4 - Moderate Structural Defect", "crack_length": "72 mm", "description": "Notable trailing edge crack. Elevated vibration in guide vane and bearings.", "recommendation": "Prepare replacement runner / perform high-priority weld repair to avoid fatigue escalation."},
    5: {"stage": "Stage 5 - Advanced Fatigue Crack", "crack_length": "95 mm", "description": "Significant crack growth across runner blade width. Risk of localized flow instability.", "recommendation": "Restrict operation at high-head resonance conditions. Expedite repair schedule."},
    6: {"stage": "Stage 6 - Critical Blade Crack", "crack_length": "115 mm", "description": "Deep structural crack. High mechanical stress concentration at trailing edge.", "recommendation": "Reduce load to limit fatigue stress. Schedule immediate offline overhaul."},
    7: {"stage": "Stage 7 - Pre-Detachment Fracture Stage", "crack_length": "140 mm", "description": "Crack approaching critical length near blade margin. Fracture imminent.", "recommendation": "Prepare for immediate controlled shutdown to prevent catastrophic fragment release."},
    8: {"stage": "Stage 8 - Detached Runner Fragment", "crack_length": "170 mm (Fragment Detached)", "description": "Blade fragment has detached. Significant hydraulic unbalance and pressure pulses.", "recommendation": "EMERGENCY SHUTDOWN REQUIRED. Inspect draft tube and tailrace for detached fragments."},
    9: {"stage": "Stage 9 - Enlarged Fragment Opening", "crack_length": "Enlarged Fracture Opening", "description": "Severe runner damage with secondary erosion and enlarged fragment opening.", "recommendation": "CRITICAL SHUTDOWN & COMPLETE RUNNER OVERHAUL. Perform full internal casing damage assessment."}
}

def get_expert_interpretation(fault_stage, api_key=None):
    """
    Query Groq LLaMA-3 model for expert interpretation, or fallback to domain static knowledge base if key is absent or API fails.
    """
    if not api_key:
        api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        fallback = STAGE_FALLBACK_INFO.get(fault_stage, STAGE_FALLBACK_INFO[1])
        fallback["source"] = "Offline Knowledge Base (Set GROQ_API_KEY for dynamic AI analysis)"
        return fallback

    try:
        client = Groq(api_key=api_key)
        prompt = f"""
You are a hydroelectric turbine condition monitoring expert. A Francis-99 turbine runner is predicted to be in fault stage `{fault_stage}` based on sensor data.

Reference stage mapping:
- Stage 0-7: Crack growth stage (Stage 1: ~15mm, Stage 2: ~30mm, Stage 3: ~50mm, Stage 4: ~72mm, Stage 5: ~95mm, Stage 6: ~115mm, Stage 7: ~140mm)
- Stage 8: Detached fragment (~170mm crack detachment)
- Stage 9: Enlarged opening of detached fragment

Provide a concise, JSON response with these keys:
{{
  "stage": "Stage name and title",
  "crack_length": "Approximate crack length in mm",
  "description": "Engineering explanation of this fault condition",
  "recommendation": "Actionable engineering advice"
}}
"""
        response = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=350,
            temperature=0.3,
        )
        content = response.choices[0].message.content
        # Extract json block if needed
        if "{" in content and "}" in content:
            json_str = content[content.find("{"):content.rfind("}")+1]
            data = json.loads(json_str)
            data["source"] = "Groq LLaMA 3 70B AI Model"
            return data
        else:
            fallback = STAGE_FALLBACK_INFO.get(fault_stage, STAGE_FALLBACK_INFO[1])
            fallback["source"] = "Offline Knowledge Base"
            return fallback
    except Exception as e:
        fallback = STAGE_FALLBACK_INFO.get(fault_stage, STAGE_FALLBACK_INFO[1])
        fallback["source"] = f"Offline Fallback (Groq API Note: {str(e)[:100]})"
        return fallback
