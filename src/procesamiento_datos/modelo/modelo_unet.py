import torch
import torch.nn as nn
import dagster as dg

class DobleConv(nn.Module):
    def __init__(self, entrada, salida, nombre=""):
        super().__init__()
        self.nombre = nombre
        self.entrada = entrada
        self.salida = salida
        
        self.doble_conv = nn.Sequential(
            nn.Conv2d(entrada, salida, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),  # ← SIN BatchNorm
            nn.Conv2d(salida, salida, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)   # ← SIN BatchNorm
            # ❌ NO Dropout aquí
        )
    
    def forward(self, x):
        # Loggear cuando se ejecuta esta convolución
        if hasattr(self, 'logger'):
            self.logger.log(f"🔄 Ejecutando DobleConv: {self.nombre} ({self.entrada} → {self.salida} canales)")
        return self.doble_conv(x)

class UNet(nn.Module):
    def __init__(self, entrada=3, salida=1, logger=None):
        super().__init__()
        self.logger = logger
        
        if self.logger:
            self.logger.log(f"🏗️ Construyendo U-Net con {entrada} canales de entrada y {salida} de salida")
        
        # Encoder
        self.enc1 = DobleConv(entrada, 64, nombre="Enc1 (3→64)")
        self.enc2 = DobleConv(64, 128, nombre="Enc2 (64→128)")
        self.enc3 = DobleConv(128, 256, nombre="Enc3 (128→256)")
        self.enc4 = DobleConv(256, 512, nombre="Enc4 (256→512)")
        
        self.pool = nn.MaxPool2d(2)
        
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # Bottleneck
        #self.bottleneck = DobleConv(256, 512, nombre="Bottleneck (256→512)")
        
        # Decoder
        #self.upconv4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec4 = DobleConv(256 + 512, 256, nombre="Dec4 (768→256)")
        #self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = DobleConv(256 + 128, 128, nombre="Dec3 (384→128)")
        #self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = DobleConv(128 + 64, 64, nombre="Dec2 (192→64)")
        #self.upconv1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        #self.dec1 = DobleConv(64, 32, nombre="Dec1 (64→32)")
        
        self.final = nn.Conv2d(64, salida, kernel_size=1)
        
        # Inyectar logger en las capas
        for module in self.modules():
            if hasattr(module, 'logger'):
                module.logger = logger
    
    def forward(self, x):
        if self.logger:
            self.logger.log(f"📊 Forward pass - Tamaño entrada: {x.shape}")
        
        # Encoder
        e1 = self.enc1(x)
        if self.logger:
            self.logger.log(f"   ✅ e1 completado: {e1.shape}")
        
        e2 = self.enc2(self.pool(e1))
        if self.logger:
            self.logger.log(f"   ✅ e2 completado: {e2.shape}")
        
        e3 = self.enc3(self.pool(e2))
        if self.logger:
            self.logger.log(f"   ✅ e3 completado: {e3.shape}")
        
        e4 = self.enc4(self.pool(e3))
        if self.logger:
            self.logger.log(f"   ✅ e4 completado: {e4.shape}")
        

        x = self.upsample(e4)
        x = torch.cat([x, e3], dim=1)
        x = self.dec4(x)

        x = self.upsample(x)
        x = torch.cat([x, e2], dim=1)
        x = self.dec3(x)

        x = self.upsample(x)
        x = torch.cat([x, e1], dim=1)
        x = self.dec2(x)

        # Bottleneck
        #b = self.bottleneck(self.pool(e4))
        #if self.logger:
        #    self.logger.log(f"   ✅ Bottleneck completado: {b.shape}")
        
        # Decoder
        #d4 = self.upconv4(b)
        #d4 = torch.cat([d4, e4], dim=1)
        #d4 = self.dec4(d4)
        #if self.logger:
        #    self.logger.log(f"   ✅ d4 completado: {d4.shape}")
        
        #d3 = self.upconv3(d4)
        #d3 = torch.cat([d3, e3], dim=1)
        #d3 = self.dec3(d3)
        #if self.logger:
        #    self.logger.log(f"   ✅ d3 completado: {d3.shape}")
        
        #d2 = self.upconv2(d3)
        #d2 = torch.cat([d2, e2], dim=1)
        #d2 = self.dec2(d2)
        #if self.logger:
        #    self.logger.log(f"   ✅ d2 completado: {d2.shape}")
        
        #d1 = self.upconv1(d2)
        #d1 = torch.cat([d1, e1], dim=1)
        #d1 = self.dec1(d1)
        #if self.logger:
        #    self.logger.log(f"   ✅ d1 completado: {d1.shape}")
        
        salida = self.final(x)
        if self.logger:
            self.logger.log(f"🎯 Forward completado - Tamaño salida: {salida.shape}")
        
        return salida