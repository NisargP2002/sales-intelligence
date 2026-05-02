from pydantic import BaseModel, Field
from typing import Optional


class ForecastRequest(BaseModel):
    days: int = Field(default=90, ge=1, le=365)
    region: Optional[str] = Field(
        default=None,
        description="Filter by region: West, East, Central, South"
    )
    category: Optional[str] = Field(
        default=None,
        description="Filter by category: Technology, Furniture, Office Supplies"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "days": 90,
                "region": None,
                "category": None
            }
        }


class ForecastPoint(BaseModel):
    date:     str
    forecast: float


class ForecastResponse(BaseModel):
    period_start:  str
    period_end:    str
    total_forecast: float
    daily_avg:     float
    points:        list[ForecastPoint]


class SummaryResponse(BaseModel):
    total_revenue:    float
    total_profit:     float
    total_orders:     int
    avg_daily_revenue: float
    best_region:      str
    best_category:    str
    date_range:       str


class RegionRevenueItem(BaseModel):
    region:  str
    revenue: float
    profit:  float


class CategoryRevenueItem(BaseModel):
    category: str
    revenue:  float
    margin:   float